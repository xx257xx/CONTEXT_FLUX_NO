from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import pack, unpack
from jaxtyping import Array, Float, PRNGKeyArray

from context_flux_no.nn.channelwise import ChannelwiseMLP
from context_flux_no.nn.hypernetwork import HypernetworkHead
from context_flux_no.nn.operators.fourier_utils import append_grid_channels

from ..abstract import AbstractMultiphysicsOperator
from .encoders import AbstractEncoder
from .target_networks import AbstractTargetNetwork
from .utils import make_encoder, make_target_network


class HyperNeuralOperator(AbstractMultiphysicsOperator):
    context_encoder: AbstractEncoder
    hypernetwork_trunk: eqx.nn.MLP
    hypernetwork_head: HypernetworkHead[AbstractTargetNetwork]
    lift_operator: eqx.nn.Identity | ChannelwiseMLP
    project_operator: eqx.nn.Identity | ChannelwiseMLP

    num_spatial_dims: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)
    boundary_condition: Literal["periodic"] = eqx.field(static=True)
    lift_dim: int | None = eqx.field(static=True)
    stack_grid: bool = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        in_timesteps: int | None,
        embedding_dim: int,
        encoder_type: Literal["ViT", "DPOT", "TRecViT"],
        encoder_kwargs: dict[str, Any],
        target_network_type: Literal["UNet", "FNO", "FluxNO"],
        target_network_kwargs: dict[str, Any],
        width_hyper: int = 128,
        depth_hyper: int = 1,
        blocks_hyper: int = 8,
        hypernet_init: Literal["default", "bias-hyperinit"] = "default",
        lift_dim: int | None = None,
        depth_lift_project: int = 1,
        activation: Callable = jax.nn.gelu,
        stack_grid: bool = True,
        boundary_condition: Literal["periodic"] = "periodic",
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        self.boundary_condition = boundary_condition

        keys = jax.random.split(key, 5)

        self.context_encoder = make_encoder(
            encoder_type,
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels + num_spatial_dims if stack_grid else in_channels,
            embedding_dim=embedding_dim,
            in_timesteps=in_timesteps,
            key=keys[0],
            **encoder_kwargs,
        )

        self.hypernetwork_trunk = eqx.nn.MLP(
            in_size=embedding_dim,
            out_size=embedding_dim,
            width_size=width_hyper,
            depth=depth_hyper,
            activation=activation,
            key=keys[1],
        )

        if lift_dim is not None:
            # Use explicit lift / project operators
            # FluxNO timestepping is done in the lifted space
            self.lift_operator = ChannelwiseMLP(
                num_spatial_dims=num_spatial_dims,
                in_channels=in_channels,
                out_channels=lift_dim,
                hidden_channels=lift_dim,
                depth=depth_lift_project,
                activation=activation,
                key=keys[2],
            )
            self.project_operator = ChannelwiseMLP(
                num_spatial_dims=num_spatial_dims,
                in_channels=lift_dim,
                out_channels=in_channels,
                hidden_channels=lift_dim,
                depth=depth_lift_project,
                activation=activation,
                key=keys[3],
            )

            target_network_init = make_target_network(
                target_network_type,
                num_spatial_dims=num_spatial_dims,
                in_channels=lift_dim,
                out_channels=lift_dim,
                key=keys[4],
                **target_network_kwargs,
            )
        else:
            self.lift_operator = eqx.nn.Identity()
            self.project_operator = eqx.nn.Identity()

            target_network_init = make_target_network(
                target_network_type,
                num_spatial_dims=num_spatial_dims,
                in_channels=in_channels,
                out_channels=in_channels,
                key=keys[4],
                **target_network_kwargs,
            )

        self.hypernetwork_head = HypernetworkHead(
            in_size=embedding_dim,
            target_network=target_network_init,
            num_blocks=blocks_hyper,
            initialization=hypernet_init,
            key=keys[3],
        )

        self.num_spatial_dims = num_spatial_dims
        self.stack_grid = stack_grid
        self.embedding_dim = embedding_dim
        self.activation = activation
        self.lift_dim = lift_dim

    def __call__(
        self,
        u: Float[Array, "time channels *grids"],
        args: tuple[float, float],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ):
        v: Float[Array, "time channels+num_spatial_dims *grids"] = jax.vmap(
            append_grid_channels
        )(u)

        context_embed: Float[Array, " embedding_dim"] = self.context_encoder(v, key=key)
        context_embed = self.hypernetwork_trunk(context_embed)
        target_network = self.hypernetwork_head(context_embed)

        u0: Float[Array, " channels *grids"] = u[-1]
        v0 = self.lift_operator(u0)
        v1: Float[Array, " channels *grids"] = target_network(v0, args)
        u1 = self.project_operator(v1)
        return u1, None


class FluxModel(eqx.Module):
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
            activation=jax.nn.gelu,
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
        a, b = self.stencil_widths
        pad_widths = [(0, 0), (a + 1, b)]
        # Need to change mode if not periodic boundary condition
        u_padded = jnp.pad(u, pad_widths, mode="wrap")
        u_stencils: Float[Array, "lift_dim grids_x+1"] = self.lift_layer(u_padded)
        f: Float[Array, "out_channels grids_x+1"] = eqx.filter_vmap(
            self.mlp, in_axes=-1, out_axes=-1
        )(u_stencils)
        return f


class HyperFluxFNOLocal(AbstractMultiphysicsOperator):
    context_encoder: AbstractEncoder
    hypernetwork_trunk: eqx.nn.MLP
    hypernetwork_heads: tuple[HypernetworkHead[FluxModel], ...]

    num_spatial_dims: int = eqx.field(static=True)
    lift_dim: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)
    stencil_size: tuple[int, int] = eqx.field(static=True)
    boundary_condition: Literal["periodic"] = eqx.field(static=True)
    stack_grid: bool = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        in_timesteps: int | None,
        embedding_dim: int,
        encoder_type: Literal["ViT", "DPOT", "TRecViT"],
        encoder_kwargs: dict[str, Any],
        depth: int,
        lift_dim: int,
        stencil_size: int | tuple[int, int],
        width_flux: int = 128,
        width_hyper: int = 128,
        depth_hyper: int = 1,
        blocks_hyper: int = 8,
        hypernet_init: Literal["default", "bias-hyperinit"] = "default",
        activation: Callable = jax.nn.gelu,
        stack_grid: bool = True,
        boundary_condition: Literal["periodic"] = "periodic",
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        self.stencil_size = (
            (stencil_size, stencil_size)
            if isinstance(stencil_size, int)
            else stencil_size
        )
        self.boundary_condition = boundary_condition

        keys = jax.random.split(key, 3)

        self.context_encoder = make_encoder(
            encoder_type,
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels + num_spatial_dims if stack_grid else in_channels,
            embedding_dim=embedding_dim,
            in_timesteps=in_timesteps,
            key=keys[0],
            **encoder_kwargs,
        )

        self.hypernetwork_trunk = eqx.nn.MLP(
            in_size=embedding_dim,
            out_size=embedding_dim,
            width_size=width_hyper,
            depth=depth_hyper,
            activation=activation,
            key=keys[1],
        )

        hypernetwork_heads = []
        for i in range(num_spatial_dims):  # Need a flux model per spatial dimension
            key_f, key_h = jax.random.split(jax.random.fold_in(keys[2], i))
            _flux_model = FluxModel(
                in_channels=in_channels + num_spatial_dims,
                out_channels=in_channels,
                stencil_widths=self.stencil_size,
                lift_dim=lift_dim,
                hidden_dim=width_flux,
                depth=depth,
                dtype=dtype,
                key=key_f,
            )
            hypernetwork_heads.append(
                HypernetworkHead(
                    in_size=embedding_dim,
                    target_network=_flux_model,
                    num_blocks=blocks_hyper,
                    initialization=hypernet_init,
                    key=key_h,
                )
            )
        self.hypernetwork_heads = tuple(hypernetwork_heads)

        self.num_spatial_dims = num_spatial_dims
        self.stack_grid = stack_grid
        self.lift_dim = lift_dim
        self.embedding_dim = embedding_dim
        self.activation = activation

    def __call__(
        self,
        u: Float[Array, "time channels *grids"],
        args: tuple[float, float],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ):
        dt, *dxs = args

        v: Float[Array, "time channels+num_spatial_dims *grids"] = jax.vmap(
            append_grid_channels
        )(u)

        context_embed: Float[Array, " embedding_dim"] = self.context_encoder(v, key=key)
        context_embed = self.hypernetwork_trunk(context_embed)

        u0: Float[Array, " channels *grids"] = u[-1]
        # Add flux for each spatial dimension
        for i, (hypernet_head, dx) in enumerate(zip(self.hypernetwork_heads, dxs)):
            flux_model = hypernet_head(context_embed)
            df = self._apply_flux(flux_model, v[-1], i)

            u0 = u0 - dt * df / dx

        return u0, None

    def _apply_flux(
        self, flux, v: Float[Array, " in_channels *grids"], spatial_axis: int
    ) -> Float[Array, " out_channels *grids"]:
        v = jnp.swapaxes(v, spatial_axis + 1, 1)
        v_, ps = pack([v], "C S *")
        f_ = eqx.filter_vmap(flux, in_axes=-1, out_axes=-1)(v_)
        f = unpack(f_, ps, "C S *")[0]
        df = jnp.diff(f, axis=1)
        df = jnp.swapaxes(df, spatial_axis + 1, 1)
        return df
