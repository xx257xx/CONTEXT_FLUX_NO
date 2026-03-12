from collections.abc import Callable
from typing import ClassVar, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from context_flux_no.models.fno import FNO1D

from ...nn import TransformerEncoderBlock
from ...nn.embedding import PatchEmbedding
from ...nn.hypernetwork import HypernetworkHead
from ...nn.misc import maybe_split
from ...nn.operators.fourier_utils import append_grid_channels
from ...nn.position_encoding import SineCosinePosEncoding2D
from .abstract import AbstractMultiphysicsOperator


# TODO: Make the model work for num_spatial_dims > 1
class ViTContextModule(eqx.Module):
    patch_embedding: PatchEmbedding
    positional_encoding: SineCosinePosEncoding2D
    encoder_blocks: list[TransformerEncoderBlock]
    layernorm: eqx.nn.LayerNorm
    num_layers: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)

    def __init__(
        self,
        patch_size: tuple[int, int],
        in_channels: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        num_hidden_patch: int,
        activation: Callable,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, num_layers + 3)
        self.patch_embedding = PatchEmbedding(
            num_spatial_dims=2,
            patch_size=patch_size,
            in_dim=in_channels,
            embedding_dim=embedding_dim,
            num_hidden=num_hidden_patch,
            hidden_dim=embedding_dim,
            activation=activation,
            dtype=dtype,
            key=keys[0],
        )
        self.positional_encoding = SineCosinePosEncoding2D(embedding_dim, key=keys[1])
        self.encoder_blocks = [
            TransformerEncoderBlock(
                num_heads,
                embedding_dim,
                hidden_dim,
                dropout,
                key=keys[i + 3],
            )
            for i in range(num_layers)
        ]
        self.layernorm = eqx.nn.LayerNorm(embedding_dim)

        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

    def __call__(
        self,
        u: Float[Array, "time channels grid_x"],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Float[Array, " embedding_dim"]:
        u = rearrange(u, "t c x -> c t x")
        u_embed: Float[Array, "embedding_dim patch_time patches_x"] = (
            self.positional_encoding(self.patch_embedding(u))
        )

        x_embed = rearrange(u_embed, "embed t x-> (t x) embed")

        keys = maybe_split(key, self.num_layers)
        for k, encoder_block in zip(keys, self.encoder_blocks):
            x_embed = encoder_block(x_embed, key=k)

        x_embed = jax.vmap(self.layernorm)(x_embed)

        return jnp.mean(x_embed, axis=0)


class HyperFluxFNO(AbstractMultiphysicsOperator):
    context_module: ViTContextModule
    hypernetwork_trunk: eqx.nn.MLP
    hypernetwork_head: HypernetworkHead[FNO1D]

    num_spatial_dims: ClassVar[int] = 1
    lift_dim: int = eqx.field(static=True)
    context_embed_dim: int = eqx.field(static=True)
    stencil_size: tuple[int, int] = eqx.field(static=True)
    boundary_condition: Literal["periodic"] = eqx.field(static=True)
    stack_grid: bool = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        data_dim: int,
        depth: int,
        frequency_modes: int,
        lift_dim: int,
        context_embed_dim: int,
        stencil_size: int | tuple[int, int],
        boundary_condition: Literal["periodic"] = "periodic",
        patch_size: tuple[int, int] = (5, 25),
        width_lift: int = 128,
        width_project: int = 128,
        width_vit: int = 128,
        width_hyper: int = 128,
        depth_lift: int = 1,
        depth_project: int = 1,
        depth_vit: int = 4,
        depth_hyper: int = 1,
        vit_heads: int = 4,
        dropout: float = 0.0,
        num_hidden_patch: int = 0,
        activation: Callable = jax.nn.gelu,
        stack_grid: bool = True,
        hypernet_init: Literal["default", "bias-hyperinit"] = "default",
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

        keys = jax.random.split(key, 4)

        input_dim = (data_dim + 1) * (self.stencil_size[0] + self.stencil_size[1] + 1)
        # input_dim = data_dim + 1
        self.hypernetwork_trunk = eqx.nn.MLP(
            in_size=context_embed_dim,
            out_size=context_embed_dim,
            width_size=width_hyper,
            depth=depth_hyper,
            activation=activation,
            key=keys[0],
        )

        flux_model = FNO1D(
            input_dim=input_dim,
            lift_dim=lift_dim,
            depth=depth,
            frequency_modes=frequency_modes,
            output_dim=data_dim,
            width_lift=width_lift,
            width_project=width_project,
            activation=activation,
            residual_connection=True,
            dtype=dtype,
            key=keys[1],
        )
        self.hypernetwork_head = HypernetworkHead(
            in_size=context_embed_dim,
            target_network=flux_model,
            initialization=hypernet_init,
            key=keys[2],
        )

        self.context_module = ViTContextModule(
            patch_size=patch_size,
            in_channels=data_dim + 1,
            embedding_dim=context_embed_dim,
            hidden_dim=width_vit,
            num_heads=vit_heads,
            num_layers=depth_vit,
            dropout=dropout,
            num_hidden_patch=num_hidden_patch,
            activation=activation,
            key=keys[3],
        )

        self.stack_grid = stack_grid
        self.lift_dim = lift_dim
        self.context_embed_dim = context_embed_dim
        self.activation = activation

    def create_stencil_axis(
        self,
        v: Float[Array, "dim x"],
    ) -> Float[Array, "data_dim {self.stencil_size[0]}+{self.stencil_size[1]}+2 grids"]:
        assert self.boundary_condition == "periodic", (
            "Other types of boundary conditions are not supported."
        )
        p, q = self.stencil_size
        v_padded: Float[Array, "in_channels grids+p+q+2"] = jnp.concatenate(
            [v[:, -p - 1 :], v, v[:, 0:q]],
            axis=1,
        )
        return jnp.stack(
            [
                jax.lax.dynamic_slice_in_dim(v_padded, i, p + q + 2, axis=-1)
                for i in range(v.shape[1])
            ],
            axis=-1,
        )

    def __call__(
        self,
        u: Float[Array, "time channels grid_x"],
        args: tuple[float, float],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ):
        dt, dx = args

        v: Float[Array, "time channels+1 grid_x"] = jax.vmap(append_grid_channels)(u)

        context_embed = self.context_module(v, key=key)
        context_embed = self.hypernetwork_trunk(context_embed)
        flux_model = self.hypernetwork_head(context_embed)

        u0 = u[-1]
        v0_stencil: Float[
            Array,
            "dim {self.stencil_size[0]}+{self.stencil_size[1]}+2 x",
        ] = self.create_stencil_axis(v[-1])

        v_l = rearrange(v0_stencil[:, :-1], "dim stencil x -> (dim stencil) x")
        v_r = rearrange(v0_stencil[:, 1:], "dim stencil x -> (dim stencil) x")
        # v_l = rearrange(v0_stencil[:, :-1], "dim stencil x -> stencil dim x")
        # v_r = rearrange(v0_stencil[:, 1:], "dim stencil x -> stencil dim x")

        # f_l = eqx.filter_vmap(flux_model, in_axes=-1, out_axes=-1)
        return u0 + dt * (flux_model(v_l) - flux_model(v_r)) / dx, None


class HyperFluxFNOLocal(AbstractMultiphysicsOperator):
    context_module: ViTContextModule
    hypernetwork_trunk: eqx.nn.MLP
    hypernetwork_head: HypernetworkHead[FNO1D]

    num_spatial_dims: ClassVar[int] = 1
    lift_dim: int = eqx.field(static=True)
    context_embed_dim: int = eqx.field(static=True)
    stencil_size: tuple[int, int] = eqx.field(static=True)
    boundary_condition: Literal["periodic"] = eqx.field(static=True)
    stack_grid: bool = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        data_dim: int,
        depth: int,
        frequency_modes: int,
        lift_dim: int,
        context_embed_dim: int,
        stencil_size: int | tuple[int, int],
        boundary_condition: Literal["periodic"] = "periodic",
        patch_size: tuple[int, int] = (5, 25),
        width_lift: int = 128,
        width_project: int = 128,
        width_vit: int = 128,
        width_hyper: int = 128,
        depth_lift: int = 1,
        depth_project: int = 1,
        depth_vit: int = 4,
        depth_hyper: int = 1,
        vit_heads: int = 4,
        dropout: float = 0.0,
        num_hidden_patch: int = 0,
        activation: Callable = jax.nn.gelu,
        stack_grid: bool = True,
        hypernet_init: Literal["default", "bias-hyperinit"] = "default",
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

        keys = jax.random.split(key, 4)

        # input_dim = (data_dim + 1) * (self.stencil_size[0] + self.stencil_size[1] + 1)
        input_dim = data_dim + 1
        self.hypernetwork_trunk = eqx.nn.MLP(
            in_size=context_embed_dim,
            out_size=context_embed_dim,
            width_size=width_hyper,
            depth=depth_hyper,
            activation=activation,
            key=keys[0],
        )

        flux_model = FNO1D(
            input_dim=input_dim,
            lift_dim=lift_dim,
            depth=depth,
            frequency_modes=frequency_modes,
            output_dim=data_dim,
            width_lift=width_lift,
            width_project=width_project,
            activation=activation,
            residual_connection=True,
            dtype=dtype,
            key=keys[1],
        )
        self.hypernetwork_head = HypernetworkHead(
            in_size=context_embed_dim,
            target_network=flux_model,
            initialization=hypernet_init,
            key=keys[2],
        )

        self.context_module = ViTContextModule(
            patch_size=patch_size,
            in_channels=data_dim + 1,
            embedding_dim=context_embed_dim,
            hidden_dim=width_vit,
            num_heads=vit_heads,
            num_layers=depth_vit,
            dropout=dropout,
            num_hidden_patch=num_hidden_patch,
            activation=activation,
            key=keys[3],
        )

        self.stack_grid = stack_grid
        self.lift_dim = lift_dim
        self.context_embed_dim = context_embed_dim
        self.activation = activation

    def create_stencil_axis(
        self,
        v: Float[Array, "dim x"],
    ) -> Float[Array, "data_dim {self.stencil_size[0]}+{self.stencil_size[1]}+2 grids"]:
        assert self.boundary_condition == "periodic", (
            "Other types of boundary conditions are not supported."
        )
        p, q = self.stencil_size
        v_padded: Float[Array, "in_channels grids+p+q+2"] = jnp.concatenate(
            [v[:, -p - 1 :], v, v[:, 0:q]],
            axis=1,
        )
        return jnp.stack(
            [
                jax.lax.dynamic_slice_in_dim(v_padded, i, p + q + 2, axis=-1)
                for i in range(v.shape[1])
            ],
            axis=-1,
        )

    def __call__(
        self,
        u: Float[Array, "time channels grid_x"],
        args: tuple[float, float],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ):
        dt, dx = args

        v: Float[Array, "time channels+1 grid_x"] = jax.vmap(append_grid_channels)(u)

        context_embed = self.context_module(v, key=key)
        context_embed = self.hypernetwork_trunk(context_embed)
        flux_model = self.hypernetwork_head(context_embed)

        u0 = u[-1]
        v0_stencil: Float[
            Array,
            "dim {self.stencil_size[0]}+{self.stencil_size[1]}+2 x",
        ] = self.create_stencil_axis(v[-1])

        # v_l = rearrange(v0_stencil[:, :-1], "dim stencil x -> (dim stencil) x")
        # v_r = rearrange(v0_stencil[:, 1:], "dim stencil x -> (dim stencil) x")
        v_l: Float[Array, "dim stencil x"] = v0_stencil[:, :-1]
        v_r: Float[Array, "dim stencil x"] = v0_stencil[:, 1:]

        f_l = jnp.mean(
            eqx.filter_vmap(flux_model, in_axes=-1, out_axes=-1)(v_l), axis=1
        )
        f_r = jnp.mean(
            eqx.filter_vmap(flux_model, in_axes=-1, out_axes=-1)(v_r), axis=1
        )
        return u0 + dt * (f_l - f_r) / dx, None
