from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import pack, rearrange, unpack
from jaxtyping import Array, Float, PRNGKeyArray

from context_flux_no.models.fno import FNO
from context_flux_no.nn.hypernetwork import HypernetworkHead
from context_flux_no.nn.operators.fourier_utils import append_grid_channels

from ..abstract import AbstractMultiphysicsOperator
from .encoders import AbstractEncoder, make_encoder


class HyperFluxFNO(AbstractMultiphysicsOperator):
    context_encoder: AbstractEncoder
    hypernetwork_trunk: eqx.nn.MLP
    hypernetwork_heads: tuple[HypernetworkHead[FNO], ...]

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
        frequency_modes: int,
        lift_dim: int,
        stencil_size: int | tuple[int, int],
        width_lift: int = 128,
        width_project: int = 128,
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

        in_channels_flux = (in_channels + num_spatial_dims) * (
            self.stencil_size[0] + self.stencil_size[1] + 1
        )

        hypernetwork_heads = []
        for i in range(num_spatial_dims):  # Need a flux model per spatial dimension
            key_f, key_h = jax.random.split(jax.random.fold_in(keys[2], i))
            _flux_model = FNO(
                num_spatial_dims=num_spatial_dims,
                in_channels=in_channels_flux,
                lift_dim=lift_dim,
                depth=depth,
                frequency_modes=frequency_modes,
                out_channels=in_channels,
                width_lift=width_lift,
                width_project=width_project,
                activation=activation,
                residual_connection=True,
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

    def create_stencil_axis(
        self, v: Float[Array, " channels *grids"], axis: int
    ) -> Float[
        Array, "data_dim {self.stencil_size[0]}+{self.stencil_size[1]}+2 *grids"
    ]:
        assert self.boundary_condition == "periodic", (
            "Other types of boundary conditions are not supported."
        )
        assert axis != 0, "axis=0 corresponds to the channel axis."
        v_ = jnp.swapaxes(v, axis, -1)  # Move target axis to 1
        p, q = self.stencil_size
        v_padded: Float[Array, " channels grids[axis]+p+q+2 *grids_rest"] = (
            jnp.concatenate(
                [v_[..., -p - 1 :], v_, v_[..., 0:q]],
                axis=-1,
            )
        )
        v_out_ = jnp.stack(
            [
                jax.lax.dynamic_slice_in_dim(v_padded, i, p + q + 2, axis=-1)
                for i in range(v_.shape[-1])
            ],
            axis=-1,
        )
        v_out_ = rearrange(
            v_out_,
            "channels ... stencil target_axis -> channels stencil ... target_axis",
        )
        return jnp.swapaxes(v_out_, axis + 1, -1)

    def __call__(
        self,
        u: Float[Array, "time channels *grids"],
        args: tuple[float, float],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ):
        dt, *dxs = args
        assert len(dxs) == self.num_spatial_dims, (
            """dx for all spatial dimensions must be provided"""
        )

        v: Float[Array, "time channels+num_spatial_dims *grids"] = jax.vmap(
            append_grid_channels
        )(u)

        context_embed: Float[Array, " embedding_dim"] = self.context_encoder(v, key=key)
        context_embed = self.hypernetwork_trunk(context_embed)

        u0: Float[Array, " channels *grids"] = u[-1]
        # Add flux for each spatial dimension
        for i, (hypernet_head, dx) in enumerate(zip(self.hypernetwork_heads, dxs)):
            v0_stencil: Float[
                Array,
                "dim {self.stencil_size[0]}+{self.stencil_size[1]}+2 *grids",
            ] = self.create_stencil_axis(v[-1], axis=i + 1)
            flux_model = hypernet_head(context_embed)

            v_l = rearrange(v0_stencil[:, :-1], "dim stencil ... -> (dim stencil) ...")
            v_r = rearrange(v0_stencil[:, 1:], "dim stencil ... -> (dim stencil) ...")

            u0 = u0 + dt * (flux_model(v_l) - flux_model(v_r)) / dx

        return u0, None


class HyperFluxFNOLocal(AbstractMultiphysicsOperator):
    context_encoder: AbstractEncoder
    hypernetwork_trunk: eqx.nn.MLP
    hypernetwork_heads: tuple[HypernetworkHead[FNO], ...]

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
        frequency_modes: int,
        lift_dim: int,
        stencil_size: int | tuple[int, int],
        width_lift: int = 128,
        width_project: int = 128,
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

        in_channels_flux = in_channels + num_spatial_dims

        hypernetwork_heads = []
        for i in range(num_spatial_dims):  # Need a flux model per spatial dimension
            key_f, key_h = jax.random.split(jax.random.fold_in(keys[2], i))
            _flux_model = FNO(
                num_spatial_dims=1,
                in_channels=in_channels_flux,
                lift_dim=lift_dim,
                depth=depth,
                frequency_modes=frequency_modes,
                out_channels=in_channels,
                width_lift=width_lift,
                width_project=width_project,
                activation=activation,
                residual_connection=True,
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

    def create_stencil_axis(
        self, v: Float[Array, " channels *grids"], axis: int
    ) -> Float[
        Array, "data_dim {self.stencil_size[0]}+{self.stencil_size[1]}+2 *grids"
    ]:
        assert self.boundary_condition == "periodic", (
            "Other types of boundary conditions are not supported."
        )
        assert axis != 0, "axis=0 corresponds to the channel axis."
        v_ = jnp.swapaxes(v, axis, -1)  # Move target axis to 1
        p, q = self.stencil_size
        v_padded: Float[Array, " channels grids[axis]+p+q+2 *grids_rest"] = (
            jnp.concatenate(
                [v_[..., -p - 1 :], v_, v_[..., 0:q]],
                axis=-1,
            )
        )
        v_out_ = jnp.stack(
            [
                jax.lax.dynamic_slice_in_dim(v_padded, i, p + q + 2, axis=-1)
                for i in range(v_.shape[-1])
            ],
            axis=-1,
        )
        v_out_ = rearrange(
            v_out_,
            "channels ... stencil target_axis -> channels stencil ... target_axis",
        )
        return jnp.swapaxes(v_out_, axis + 1, -1)

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
            v0_stencil: Float[
                Array,
                "dim {self.stencil_size[0]}+{self.stencil_size[1]}+2 *grids",
            ] = self.create_stencil_axis(v[-1], axis=i + 1)
            flux_model = hypernet_head(context_embed)

            v_l: Float[Array, "dim stencil *grids"] = v0_stencil[:, :-1]
            v_r: Float[Array, "dim stencil *grids"] = v0_stencil[:, 1:]

            f_l = self._apply_flux(flux_model, v_l)
            f_r = self._apply_flux(flux_model, v_r)
            u0 = u0 + dt * (f_l - f_r) / dx

        return u0, None

    def _apply_flux(
        self, flux, v: Float[Array, "in_channels stencil *grids"]
    ) -> Float[Array, " out_channels *grids"]:
        v_, ps = pack([v], "C S *")
        f_ = jnp.mean(eqx.filter_vmap(flux, in_axes=-1, out_axes=-1)(v_), axis=1)
        f = unpack(f_, ps, "C *")[0]
        return f
