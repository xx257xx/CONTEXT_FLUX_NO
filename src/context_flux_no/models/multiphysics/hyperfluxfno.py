from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from ...nn import TransformerEncoderBlock
from ...nn.embedding import PatchEmbedding
from ...nn.hypernetworks import HyperFourier, HyperLinear
from ...nn.operators.fourier_utils import append_grid_channels
from ...nn.position_encoding import SineCosinePosEncoding2D
from .abstract import AbstractMultiphysicsOperator


# TODO: replace PatchEmbedding with the new implementation
class ViTContextModule(eqx.Module):
    patch_embedding: PatchEmbedding
    positional_encoding: SineCosinePosEncoding2D
    encoder_blocks: list[TransformerEncoderBlock]
    layernorm: eqx.nn.LayerNorm
    num_layers: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        patch_size: tuple[int, int],
        in_channels: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        hidden_dim_patch: int,
        activation: Callable,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, num_layers + 3)
        self.patch_embedding = PatchEmbedding(
            num_spatial_dims=num_spatial_dims,
            patch_size=patch_size,
            in_dim=in_channels + num_spatial_dims,
            embedding_dim=embedding_dim,
            num_hidden=1,
            hidden_dim=hidden_dim_patch,
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
        u: Float[Array, "time channels *grids"],
        args: tuple[float, float],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Float[Array, "time embedding_dim, *patches"]:
        u_embed: Float[Array, "time embedding_dim, *patches"] = eqx.filter_vmap(
            self.patch_embedding
        )(u)
        u_embed = eqx.filter_vmap(self.positional_encoding)(u_embed)

        x_embed = rearrange(x_embed, "row col embed -> (row col) embed")

        keys = jax.random.split(key, self.num_layers)
        for k, encoder_block in zip(keys, self.encoder_blocks, strict=False):
            x_embed = encoder_block(x_embed, key=k)

        x_embed = jax.vmap(self.layernorm)(x_embed)

        return jnp.mean(x_embed, axis=0), rearrange(
            x_embed,
            "(row col) embed -> row col embed",
            row=row,
        )


class ViTContextHyperFluxFNO(AbstractMultiphysicsOperator):
    context_module: ViTContextModule
    hypernetwork_trunk: eqx.nn.MLP
    hyperlift_layers: list[HyperLinear]
    hyperfourier_layers: list[HyperFourier]
    hyperproject_layers: list[HyperLinear]
    film_net: eqx.nn.Linear

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
        activation: Callable = jax.nn.gelu,
        stack_grid: bool = True,
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

        keys = jax.random.split(key, 7 + depth)

        input_dim = data_dim * (self.stencil_size[0] + self.stencil_size[1] + 1)
        self.hypernetwork_trunk = eqx.nn.MLP(
            in_size=context_embed_dim,
            out_size=context_embed_dim,
            width_size=width_hyper,
            depth=depth_hyper,
            activation=activation,
            key=keys[0],
        )
        self.hyperlift_layers = [
            HyperLinear(
                in_features=input_dim + 1 if stack_grid else input_dim,
                out_features=width_lift,
                hyper_in_dims=context_embed_dim,
                dtype=dtype,
                key=keys[1],
            ),
            HyperLinear(
                in_features=width_lift,
                out_features=lift_dim,
                hyper_in_dims=context_embed_dim,
                dtype=dtype,
                key=keys[2],
            ),
        ]

        self.context_module = ViTContextModule(
            patch_size=patch_size,
            in_channels=data_dim,
            embedding_dim=context_embed_dim,
            hidden_dim=width_vit,
            num_heads=vit_heads,
            num_layers=depth_vit,
            dropout=dropout,
            key=keys[3],
        )
        self.hyperfourier_layers = [
            HyperFourier(
                num_spatial_dims=1,
                in_channels=lift_dim,
                out_channels=lift_dim,
                frequency_modes=frequency_modes,
                hyper_in_dims=context_embed_dim,
                activation=activation,
                dtype=dtype,
                key=k,
            )
            for k in keys[4 : 4 + depth]
        ]

        self.hyperproject_layers = [
            HyperLinear(
                in_features=lift_dim,
                out_features=width_project,
                hyper_in_dims=context_embed_dim,
                dtype=dtype,
                key=keys[-3],
            ),
            HyperLinear(
                in_features=width_project,
                out_features=data_dim,
                hyper_in_dims=context_embed_dim,
                dtype=dtype,
                key=keys[-2],
            ),
        ]

        self.film_net = eqx.nn.Linear(context_embed_dim, 2 * lift_dim, key=keys[-1])

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

    def lift_layer(
        self, v: Float[Array, " input_dim"], context_vec: Float[Array, " embedding_dim"]
    ) -> Float[Array, " lift_dim"]:
        v = self.hyperlift_layers[0](v, context_vec)
        v = self.activation(v)
        v = self.hyperlift_layers[1](v, context_vec)
        return v

    def project_layer(
        self, v: Float[Array, " lift_dim"], context_vec: Float[Array, " embedding_dim"]
    ) -> Float[Array, " data_dim"]:
        v = self.hyperproject_layers[0](v, context_vec)
        v = self.activation(v)
        v = self.hyperproject_layers[1](v, context_vec)
        return v

    def flux_model(
        self,
        v_stencil: Float[Array, "dim_stencil x"],
        context_vec: Float[Array, " embedding_dim"],
        film_weights: Float[Array, "2*lift_dim x"],
    ):
        """dim_stencil = dim*(self.stencil_size[0]+self.stencil_size[1]+1)"""
        if self.stack_grid:
            grid = jnp.expand_dims(jnp.linspace(0, 1, v_stencil.shape[-1]), axis=0)
            v_stencil = jnp.concatenate((v_stencil, grid), axis=0)

        # gamma, beta = film_weights[: self.lift_dim], film_weights[self.lift_dim :]
        gamma, beta = 1.0, 0.0

        v: Float[Array, "lift_dim x"] = eqx.filter_vmap(
            self.lift_layer,
            in_axes=(-1, None),
            out_axes=-1,
        )(v_stencil, context_vec)

        for hyperfourier in self.hyperfourier_layers:
            v = v + gamma * hyperfourier(v, context_vec) + beta

        v = eqx.filter_vmap(self.project_layer, in_axes=(-1, None), out_axes=-1)(
            v, context_vec
        )
        return v

    def __call__(
        self,
        u: Float[Array, "time channels *grids"],
        args: tuple[float, float],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ):
        dt, dx = args

        u: Float[Array, "time channels+num_spatial_dims *grids"] = jax.vmap(
            append_grid_channels
        )(u)

        context_embed, context_patches = self.context_module(
            rearrange(context, "t d x -> t x d"),
            key=key,
        )
        context_embed = self.hypernetwork_trunk(context_embed)
        # context_patches_x: Float[Array, "patches_x embedding_dim"] = jnp.mean(
        #     context_patches,
        #     axis=0,
        # )
        # film_weights: Float[Array, "2*lift_dim patches_x"] = jax.vmap(
        #     self.film_net,
        #     in_axes=0,
        #     out_axes=1,
        # )(context_patches_x)

        # film_weights: Float[Array, "2*lift_dim x"] = jax.image.resize(
        #     film_weights,
        #     (film_weights.shape[0], v.shape[1]),
        #     method="nearest",
        # )
        film_weights = None
        v_stencil: Float[
            Array,
            "dim {self.stencil_size[0]}+{self.stencil_size[1]}+2 x",
        ] = self.create_stencil_axis(v)

        v_l = rearrange(v_stencil[:, :-1], "dim stencil x -> (dim stencil) x")
        v_r = rearrange(v_stencil[:, 1:], "dim stencil x -> (dim stencil) x")

        f_l = self.flux_model(v_l, context_embed, film_weights)
        f_r = self.flux_model(v_r, context_embed, film_weights)
        return v + dt * (f_l - f_r) / dx
