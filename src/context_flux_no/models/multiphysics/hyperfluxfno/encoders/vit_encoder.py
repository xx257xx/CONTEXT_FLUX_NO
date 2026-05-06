from collections.abc import Callable
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from context_flux_no.nn import PatchEmbedding, TransformerEncoderBlock
from context_flux_no.nn.misc import maybe_split
from context_flux_no.nn.position_encoding import SineCosinePosEncoding2D

from .base import AbstractEncoder


# TODO: Make the model work for num_spatial_dims > 1
class ViTEncoder(AbstractEncoder):
    num_spatial_dims: ClassVar[int] = 1
    in_channels: int = eqx.field(static=True)
    in_timesteps: ClassVar[None] = None
    embedding_dim: int = eqx.field(static=True)

    patch_embedding: PatchEmbedding
    positional_encoding: SineCosinePosEncoding2D
    encoder_blocks: list[TransformerEncoderBlock]
    layernorm: eqx.nn.LayerNorm
    num_layers: int = eqx.field(static=True)

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
        self.in_channels = in_channels
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
