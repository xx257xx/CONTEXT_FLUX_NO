from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import pack, unpack
from jaxtyping import Array, Float, PRNGKeyArray

from context_flux_no.nn.operators.fourier_utils import append_grid_channels

from .base import AbstractTargetNetwork


class NeuralNetworkFlux(eqx.Module):
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    stencil_widths: tuple[int, int] = eqx.field(static=True)
    lift_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    lift_layer: eqx.nn.Conv1d
    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stencil_widths: tuple[int, int],
        lift_dim: int,
        hidden_dim: int,
        depth: int,
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, 2)
        kernel_size = stencil_widths[0] + stencil_widths[1] + 1
        self.lift_layer = eqx.nn.Conv1d(
            in_channels=in_channels,
            out_channels=lift_dim,
            kernel_size=kernel_size,
            dtype=dtype,
            key=keys[0],
        )
        self.mlp = eqx.nn.MLP(
            in_size=lift_dim,
            out_size=out_channels,
            width_size=hidden_dim,
            depth=depth,
            activation=activation,
            dtype=dtype,
            key=keys[1],
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stencil_widths = stencil_widths
        self.lift_dim = lift_dim
        self.hidden_dim = hidden_dim
        self.depth = depth

    @property
    def stencil_size(self) -> int:
        return sum(self.stencil_widths) + 1

    def __call__(
        self,
        u: Float[Array, "in_channels grids"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "out_channels grids+1"]:
        del key
        a, b = self.stencil_widths
        pad_widths = [(0, 0), (a + 1, b)]
        # Need to change mode if not periodic boundary condition
        u_padded = jnp.pad(u, pad_widths, mode="wrap")
        u_stencils: Float[Array, "lift_dim grids_x+1"] = self.lift_layer(u_padded)
        f: Float[Array, "out_channels grids_x+1"] = eqx.filter_vmap(
            self.mlp, in_axes=-1, out_axes=-1
        )(u_stencils)
        return f


class FluxNOTargetNetwork(AbstractTargetNetwork):
    num_spatial_dims: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    stack_grid: bool = eqx.field(static=True)

    fluxes: tuple[NeuralNetworkFlux]

    def __init__(self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        lift_dim: int,
        depth: int,
        stencil_widths: tuple[int, int],
        hidden_dim: int,
        stack_grid: bool = True,
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        in_channels_ = in_channels + num_spatial_dims if stack_grid else in_channels
        fluxes = [
            NeuralNetworkFlux(
                in_channels=in_channels_,
                out_channels=out_channels,
                stencil_widths=stencil_widths,
                lift_dim=lift_dim,
                hidden_dim=hidden_dim,
                depth=depth,
                activation=activation,
                dtype=dtype,
                key=k,
            )
            for k in jax.random.split(key, num_spatial_dims)
        ]
        self.fluxes = tuple(fluxes)
        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stack_grid = stack_grid

    def __call__(
        self,
        u: Float[Array, " in_channels *spatial_dims"],
        args: tuple[float, ...],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Float[Array, " out_channels *spatial_dims"]:
        del key, inference
        dt, *dxs = args

        v = append_grid_channels(u) if self.stack_grid else u
        for i, (flux_func, dx) in enumerate(zip(self.fluxes, dxs)):
            df = self.compute_flux_difference(flux_func, v, spatial_axis=i)
            u = u - dt * df / dx
        return u

    def compute_flux_difference(
        self,
        flux_func: Callable[
            [Float[Array, "in_channels x"]], Float[Array, "in_channels x+1"]
        ],
        v: Float[Array, " in_channels *spatial_dims"],
        spatial_axis: int,
    ) -> Float[Array, " out_channels *spatial_dims"]:
        v = jnp.swapaxes(v, spatial_axis + 1, 1)
        v_, ps = pack([v], "C S *")
        f_ = eqx.filter_vmap(flux_func, in_axes=-1, out_axes=-1)(v_)
        f = unpack(f_, ps, "C S *")[0]
        df = jnp.diff(f, axis=1)
        df = jnp.swapaxes(df, spatial_axis + 1, 1)
        return df
