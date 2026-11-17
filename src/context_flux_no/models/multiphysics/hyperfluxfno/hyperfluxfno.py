from collections.abc import Callable
from math import prod, sqrt
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import pack, rearrange, reduce, unpack
from equinox.nn._misc import default_init
from jaxtyping import Array, Float, PRNGKeyArray

from context_flux_no.nn.channelwise import ChannelwiseMLP
from context_flux_no.nn.hypernetwork import HypernetworkHead
from context_flux_no.nn.operators.fourier_utils import append_grid_channels

from ..abstract import AbstractGeneratedSolverOperator, AbstractMultiphysicsOperator
from .encoders import AbstractEncoder
from .target_networks import AbstractTargetNetwork
from .utils import make_encoder, make_target_network


def scale(x: Array, axis: int | tuple[int, ...], eps: float = 1e-6):
    return jnp.sqrt(jnp.mean(x**2, axis=axis) + eps)


class InterfaceReconstructor(eqx.Module):
    """Given an array of cell-averaged states (u_i, ..., u_n), reconstruct the state at
    the cell interfaces: (u_{-1/2}, ..., u_{n+1/2}). For each state u_i, this is
    accomplished by using a spatial kernel of width (a, b):
    u_{i+1/2} = F(u_{i-a}, ..., u_{i+b}).

    The reconstructor works on per-stencil normalized states:
    \tilde{u}_i=(u_i-u_{center})/u_{scale}, with u_{center}=(u_{max}+u_{min})/2 and
    u_{scale}=(u_{max}-u_{min})/2.
    The edge case of constant stencil (u_{max}\approx u_{min}) is handled separately to map to
    zero.
    """

    weight: Float[
        Array, " hidden 1 kernel_normal *kernels_transverse"
    ]  # shared across channels
    bias: Float[Array, " hidden *ones"]
    weight_out: Float[Array, "out hidden"]
    bias_out: Float[Array, " out * ones"]

    num_spatial_dims: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_per_channel: int = eqx.field(static=True)
    normal_stencil_halfwidths: tuple[int, int] = eqx.field(static=True)
    transverse_stencil_halfwidth: int = eqx.field(static=True)
    boundary_condition: Literal["periodic"] = "periodic"
    eps: float = eqx.field(static=True)
    activation: Callable

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_per_channel: int,
        hidden_channels: int,
        normal_stencil_halfwidths: tuple[int, int],
        transverse_stencil_halfwidth: int = 1,
        activation: Callable = jax.nn.gelu,
        boundary_condition: Literal["periodic"] = "periodic",
        eps: float = 1e-6,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        if any(
            w < 0 for w in (*normal_stencil_halfwidths, transverse_stencil_halfwidth)
        ):
            raise ValueError("Stencil widths must be non-negative")
        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_per_channel = out_per_channel
        self.normal_stencil_halfwidths = normal_stencil_halfwidths
        self.transverse_stencil_halfwidth = transverse_stencil_halfwidth
        self.boundary_condition = boundary_condition
        self.eps = eps
        self.activation = activation

        wkey, bkey, wokey, bokey = jax.random.split(key, 4)
        kernel_sizes = (self.normal_stencil_width,) + (
            self.transverse_stencil_width,
        ) * (self.num_spatial_dims - 1)
        ones = (1,) * self.num_spatial_dims

        lim = 1 / sqrt(prod(kernel_sizes))
        self.weight = default_init(
            wkey, (hidden_channels, 1) + kernel_sizes, dtype, lim
        )
        self.bias = default_init(bkey, (hidden_channels,) + ones, dtype, lim)

        lim_out = 1 / sqrt(hidden_channels)

        self.weight_out = default_init(
            wokey, (out_per_channel, hidden_channels), dtype, lim_out
        )
        self.bias_out = default_init(bokey, (out_per_channel,) + ones, dtype, lim_out)

    @property
    def normal_stencil_width(self) -> int:
        return sum(self.normal_stencil_halfwidths) + 1

    @property
    def transverse_stencil_width(self) -> int:
        return 2 * self.transverse_stencil_halfwidth + 1

    def pad_ghost_cells(
        self, u: Float[Array, " in_channels *grids"], normal_axis: int
    ) -> Float[Array, " in_channels *grids_padded"]:
        a, b = self.normal_stencil_halfwidths
        s = self.transverse_stencil_halfwidth

        pad = [(s, s)] * self.num_spatial_dims
        pad[normal_axis] = (a + 1, b)  # N+1 faces along the normal axis

        if self.boundary_condition == "periodic":
            return jnp.pad(u, [(0, 0)] + pad, mode="wrap")
        else:
            # In the future, support other types of BCs too.
            raise NotImplementedError(self.boundary_condition)

    def stencil_shape(self, normal_axis: int) -> tuple[int, ...]:
        """Per-axis stencil sizes when `normal_axis` is the normal direction."""
        w = [self.transverse_stencil_width] * self.num_spatial_dims
        w[normal_axis] = self.normal_stencil_width
        return tuple(w)

    def _per_stencil_shift_scale(
        self, x: Float[Array, " *grids"], normal_axis: int
    ) -> tuple[Float[Array, " *grids_out"], Float[Array, " *grids_out"]]:
        window_dims = self.stencil_shape(normal_axis)
        strides = (1,) * self.num_spatial_dims
        x_max = jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max, window_dims, strides, padding="VALID"
        )
        x_min = jax.lax.reduce_window(
            x, jnp.inf, jax.lax.min, window_dims, strides, padding="VALID"
        )
        return 0.5 * (x_max + x_min), 0.5 * (x_max - x_min)

    def _per_channel_reconstruction(
        self, x: Float[Array, " *grids_padded"], normal_axis: int
    ) -> Float[Array, " out_per_channel *faces"]:
        shift, scale = self._per_stencil_shift_scale(x, normal_axis)
        is_constant_patch = scale <= self.eps

        w_patch: Float[Array, " hidden_channels *grids_out"] = (
            jax.lax.conv_general_dilated(
                rearrange(x, "... -> 1 1 ..."),
                jnp.moveaxis(self.weight, 2, 2 + normal_axis),
                window_strides=(1,) * self.num_spatial_dims,
                padding="VALID",
            )[0]
        )
        shift_w: Float[Array, " hidden_channels *grids_out"] = jnp.einsum(
            "i,...->i...", reduce(self.weight, "C ... -> C", "sum"), shift
        )

        scale_safe = jnp.where(is_constant_patch, 1.0, scale)

        x1: Float[Array, " hidden_channels *grids_out"] = (
            jnp.where(is_constant_patch, 0.0, (w_patch - shift_w) / scale_safe)
            + self.bias
        )
        x1 = self.activation(x1)
        x2 = jnp.einsum("ij,j...->i...", self.weight_out, x1) + self.bias_out
        return scale * x2 + shift

    def __call__(self, u: Float[Array, " in_channels *grids"], normal_axis: int):
        u_ = self.pad_ghost_cells(u, normal_axis)
        u_out: Float[Array, " in_channels*out_per_channel *grids"] = jax.vmap(
            lambda x: self._per_channel_reconstruction(x, normal_axis)
        )(u_)
        return rearrange(u_out, "I O ... -> (I O) ...")


class HyperNeuralOperatorv2(AbstractMultiphysicsOperator):
    context_encoder: AbstractEncoder
    hypernetwork_trunk: eqx.nn.MLP
    hypernetwork_head: HypernetworkHead[AbstractTargetNetwork]
    interface_reconstructor: InterfaceReconstructor

    num_spatial_dims: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)
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
        normal_stencil_halfwidths: tuple[int, int],
        transverse_stencil_halfwidth: int,
        reconstructions_per_channel: int,
        depth_target: int = 1,
        width_hyper: int = 128,
        depth_hyper: int = 1,
        blocks_hyper: int = 8,
        hidden_dim: int = 64,
        hypernet_init: Literal["default", "bias-hyperinit"] = "default",
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
            in_size=embedding_dim + in_channels,
            out_size=embedding_dim,
            width_size=width_hyper,
            depth=depth_hyper,
            activation=activation,
            key=keys[1],
        )

        self.interface_reconstructor = InterfaceReconstructor(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_per_channel=reconstructions_per_channel,
            hidden_channels=hidden_dim,
            normal_stencil_halfwidths=normal_stencil_halfwidths,
            transverse_stencil_halfwidth=transverse_stencil_halfwidth,
            boundary_condition=boundary_condition,
            dtype=dtype,
            key=keys[2],
        )

        target_network = ChannelwiseMLP(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels * reconstructions_per_channel,
            out_channels=num_spatial_dims * in_channels,
            hidden_channels=hidden_dim,
            depth=depth_target,
            activation=activation,
            dtype=dtype,
            key=keys[3],
        )

        self.hypernetwork_head = HypernetworkHead(
            in_size=embedding_dim,
            target_network=target_network,
            num_blocks=blocks_hyper,
            initialization=hypernet_init,
            key=keys[4],
        )

        self.num_spatial_dims = num_spatial_dims
        self.stack_grid = stack_grid
        self.embedding_dim = embedding_dim
        self.activation = activation

    def __call__(
        self,
        u: Float[Array, "time channels *grids"],
        args: tuple[float, ...],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ):
        dt, *dxs = args

        if self.stack_grid:
            v: Float[Array, "time channels+num_spatial_dims *grids"] = jax.vmap(
                append_grid_channels
            )(u)
        else:
            v = u

        context_embed: Float[Array, " embedding_dim"] = self.context_encoder(v, key=key)
        context_scale = reduce(u, "T C ... -> C", scale)
        context_embed = self.hypernetwork_trunk(
            jnp.concatenate((context_embed, jnp.log(context_scale)))
        )
        target_network = self.hypernetwork_head(context_embed)

        u_scale = rearrange(context_scale, "C -> C" + " 1" * self.num_spatial_dims)
        u0: Float[Array, " channels *grids"] = u[-1] / u_scale

        def flux_divergence(_u0):
            div = jnp.zeros_like(_u0)
            for i, dx in enumerate(dxs):
                v_i = self.interface_reconstructor(_u0, normal_axis=i)
                f_i = rearrange(
                    target_network(v_i), "(d c) ... -> d c ...", d=self.num_spatial_dims
                )[i]
                div = div + jnp.diff(f_i, axis=i + 1) / dx
            return div

        u1 = (u0 - dt * flux_divergence(u0)) * u_scale
        return u1, None


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
            in_size=embedding_dim + in_channels,
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
        if self.stack_grid:
            v: Float[Array, "time channels+num_spatial_dims *grids"] = jax.vmap(
                append_grid_channels
            )(u)
        else:
            v = u

        context_embed: Float[Array, " embedding_dim"] = self.context_encoder(v, key=key)
        # Context scaling is optional. Currently have a non-scaled impl, but later will be merged into a single class with a boolean flag.
        context_scale = reduce(u, "T C ... -> C", scale)
        context_embed = self.hypernetwork_trunk(
            jnp.concatenate((context_embed, jnp.log(context_scale)))
        )
        target_network = self.hypernetwork_head(context_embed)

        u_scale = rearrange(context_scale, "C -> C" + " 1" * self.num_spatial_dims)
        u0: Float[Array, " channels *grids"] = u[-1] / u_scale
        v0 = self.lift_operator(u0)
        v1: Float[Array, " channels *grids"] = target_network(v0, args)
        u1 = self.project_operator(v1) * u_scale
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


class HyperFluxFNOLocal(AbstractGeneratedSolverOperator[tuple[FluxModel, ...]]):
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

    # TODO: `__call__` repeats the bodies of `generate_solver` and `step`, because the
    # function of the submission is not changed during the rebuttal. It should call
    # them, so that the class has one route to a step.
    def generate_solver(
        self,
        context: Float[Array, "time channels *grids"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[FluxModel, ...]:
        """Generate the flux network of every spatial dimension from a context.

        The solver of HFluxNO is the tuple of generated flux networks, one per spatial
        dimension. Action 3b of the ICLR 2027 rebuttal plan evaluates them directly.

        **Arguments:**

        - `context`: The frames that the encoder reads.
        """
        v: Float[Array, "time channels+num_spatial_dims *grids"] = jax.vmap(
            append_grid_channels
        )(context)
        context_embed: Float[Array, " embedding_dim"] = self.context_encoder(v, key=key)
        context_embed = self.hypernetwork_trunk(context_embed)
        return tuple(head(context_embed) for head in self.hypernetwork_heads)

    def step(
        self,
        solver: tuple[FluxModel, ...],
        u: Float[Array, " channels *grids"],
        args: tuple[float, ...],
    ) -> Float[Array, " channels *grids"]:
        """Advance one state by one finite-volume step with given flux networks.

        **Arguments:**

        - `solver`: The flux network of every spatial dimension, as `generate_solver`
            returns them.
        - `u`: The state to advance.
        - `args`: The time step followed by the grid spacing of every spatial dimension.
        """
        dt, *dxs = args
        v: Float[Array, " channels+num_spatial_dims *grids"] = append_grid_channels(u)
        for i, (flux_model, dx) in enumerate(zip(solver, dxs)):
            u = u - dt * self._apply_flux(flux_model, v, i) / dx
        return u

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


class ContextConditionedFluxModel(eqx.Module):
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    stencil_widths: tuple[int, int] = eqx.field(static=True)
    lift_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    context_size: int = eqx.field(static=True)

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
        context_size: int,
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
            in_size=lift_dim + context_size,
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
        self.context_size = context_size

    @property
    def stencil_size(self) -> int:
        return sum(self.stencil_widths) + 1

    def __call__(
        self,
        u: Float[Array, "in_channels grids"],
        context: Float[Array, " context_size"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "out_channels grids+1"]:
        a, b = self.stencil_widths
        pad_widths = [(0, 0), (a + 1, b)]
        # Need to change mode if not periodic boundary condition
        u_padded = jnp.pad(u, pad_widths, mode="wrap")
        u_stencils: Float[Array, "lift_dim grids_x+1"] = self.lift_layer(u_padded)

        context = jnp.broadcast_to(jnp.expand_dims(context, axis=-1), u_stencils.shape)
        u_stencils: Float[Array, "lift_dim+context_size grids_x+1"] = jnp.concatenate(
            (context, u_stencils), axis=0
        )
        f: Float[Array, "out_channels grids_x+1"] = eqx.filter_vmap(
            self.mlp, in_axes=-1, out_axes=-1
        )(u_stencils)
        return f


class ContextAppendedFluxNO(
    AbstractGeneratedSolverOperator[Float[Array, " embedding_dim"]]
):
    context_encoder: AbstractEncoder
    hypernetwork_trunk: eqx.nn.MLP
    fluxes: tuple[ContextConditionedFluxModel, ...]

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

        fluxes = []
        for i in range(num_spatial_dims):  # Need a flux model per spatial dimension
            key_f = jax.random.fold_in(keys[2], i)
            fluxes.append(
                ContextConditionedFluxModel(
                    in_channels=in_channels + num_spatial_dims,
                    out_channels=in_channels,
                    stencil_widths=self.stencil_size,
                    lift_dim=lift_dim,
                    hidden_dim=width_flux,
                    depth=depth,
                    context_size=embedding_dim,
                    dtype=dtype,
                    key=key_f,
                )
            )

        self.fluxes = tuple(fluxes)

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
        for i, (flux_model, dx) in enumerate(zip(self.fluxes, dxs)):
            df = self._apply_flux(flux_model, v[-1], context_embed, i)

            u0 = u0 - dt * df / dx

        return u0, None

    # TODO: `__call__` repeats the bodies of `generate_solver` and `step`, because the
    # function of the submission is not changed during the rebuttal. It should call
    # them, so that the class has one route to a step.
    def generate_solver(
        self,
        context: Float[Array, "time channels *grids"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " embedding_dim"]:
        """Encode a context into the vector that the flux networks are conditioned on.

        The solver of this model is the context vector after the hypernetwork trunk.
        Held fixed, it makes the flux networks a fixed solver, as the generated flux
        networks of `HyperFluxFNOLocal` are (ICLR 2027 rebuttal plan, action 4f).

        **Arguments:**

        - `context`: The frames that the encoder reads.
        """
        v: Float[Array, "time channels+num_spatial_dims *grids"] = jax.vmap(
            append_grid_channels
        )(context)
        context_embed: Float[Array, " embedding_dim"] = self.context_encoder(v, key=key)
        return self.hypernetwork_trunk(context_embed)

    def step(
        self,
        solver: Float[Array, " embedding_dim"],
        u: Float[Array, " channels *grids"],
        args: tuple[float, ...],
    ) -> Float[Array, " channels *grids"]:
        """Advance one state by one finite-volume step with a given context vector.

        **Arguments:**

        - `solver`: The context vector, as `generate_solver` returns it.
        - `u`: The state to advance.
        - `args`: The time step followed by the grid spacing of every spatial dimension.
        """
        dt, *dxs = args
        v: Float[Array, " channels+num_spatial_dims *grids"] = append_grid_channels(u)
        for i, (flux_model, dx) in enumerate(zip(self.fluxes, dxs)):
            u = u - dt * self._apply_flux(flux_model, v, solver, i) / dx
        return u

    def _apply_flux(
        self, flux, v: Float[Array, " in_channels *grids"], context, spatial_axis: int
    ) -> Float[Array, " out_channels *grids"]:
        v = jnp.swapaxes(v, spatial_axis + 1, 1)
        v_, ps = pack([v], "C S *")
        f_ = eqx.filter_vmap(flux, in_axes=(-1, None), out_axes=-1)(v_, context)
        f = unpack(f_, ps, "C S *")[0]
        df = jnp.diff(f, axis=1)
        df = jnp.swapaxes(df, spatial_axis + 1, 1)
        return df
