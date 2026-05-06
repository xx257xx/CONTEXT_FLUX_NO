from collections.abc import Callable, Sequence
from functools import partial
from math import sqrt
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from context_flux_no.nn.embedding import PatchEmbedding
from context_flux_no.nn.misc import maybe_split, to_ntuple
from context_flux_no.nn.position_encoding import LearnedPositionEncoding
from context_flux_no.nn.ssm import RG_LRU
from context_flux_no.nn.transformer import TransformerEncoderBlock

from .base import AbstractEncoder


class Tokenizer(eqx.Module):
    patch_embedding: PatchEmbedding
    # Will need fixed encoding scheme for arbitrary grid size handling
    position_encoding: LearnedPositionEncoding

    num_spatial_dims: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        grid_size: int | Sequence[int],
        patch_size: int | Sequence[int],
        in_dim: int,
        embedding_dim: int,
        num_hidden: int = 0,
        hidden_dim: int = 32,
        activation: Callable = jax.nn.gelu,
        final_activation: Callable = lambda x: x,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        key_emb, key_enc = jax.random.split(key)

        patch_size = to_ntuple(patch_size, num_spatial_dims)
        grid_size = to_ntuple(grid_size, num_spatial_dims)
        self.patch_embedding = PatchEmbedding(
            num_spatial_dims=num_spatial_dims,
            patch_size=patch_size,
            in_dim=in_dim,
            embedding_dim=embedding_dim,
            num_hidden=num_hidden,
            hidden_dim=hidden_dim,
            activation=activation,
            final_activation=final_activation,
            dtype=dtype,
            key=key_emb,
        )

        # init_scale consistent with that of https://github.com/google-deepmind/trecvit/blob/main/trecvit/utils.py
        self.position_encoding = LearnedPositionEncoding(
            channels=embedding_dim,
            spatial_dims=self.patch_embedding.output_size(grid_size),
            init_scale=1 / sqrt(embedding_dim),
            key=key_enc,
        )

        self.num_spatial_dims = num_spatial_dims
        self.embedding_dim = embedding_dim

    def __call__(
        self, u: Float[Array, " channel *grids"], *, key: PRNGKeyArray | None = None
    ) -> Float[Array, " embedding_dim *grids_patch"]:
        return self.position_encoding(self.patch_embedding(u, key=key), resize=False)


class RecurrentBlock(eqx.Module):
    channels: int = eqx.field(static=True)
    lru_width: int = eqx.field(static=True)

    conv1d: eqx.nn.Conv1d
    lru: RG_LRU
    linear_y: eqx.nn.Linear
    linear_x: eqx.nn.Linear
    linear_out: eqx.nn.Linear

    def __init__(
        self,
        channels: int,
        lru_width: int,
        conv1d_kernel_size: int,
        lru_num_heads: int,
        lru_a_init_minval: float,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, 5)
        self.linear_y = eqx.nn.Linear(
            in_features=channels, out_features=lru_width, dtype=dtype, key=keys[0]
        )
        self.linear_x = eqx.nn.Linear(
            in_features=channels, out_features=lru_width, dtype=dtype, key=keys[1]
        )

        # Set groups=lru_width for separable convolution
        self.conv1d = eqx.nn.Conv1d(
            in_channels=lru_width,
            out_channels=lru_width,
            kernel_size=conv1d_kernel_size,
            padding=[
                (conv1d_kernel_size - 1, 0),
            ],
            groups=lru_width,
            dtype=dtype,
            key=keys[2],
        )
        self.lru = RG_LRU(
            channels=lru_width,
            num_heads=lru_num_heads,
            a_init_minval=lru_a_init_minval,
            dtype=dtype,
            key=keys[3],
        )

        self.linear_out = eqx.nn.Linear(
            in_features=lru_width, out_features=channels, dtype=dtype, key=keys[4]
        )

        self.channels = channels
        self.lru_width = lru_width

    def __call__(
        self,
        u: Float[Array, "seq channels"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "seq channels"]:
        y: Float[Array, "seq lru_width"] = jax.nn.gelu(
            eqx.filter_vmap(self.linear_y)(u)
        )

        # Change dimension order here to match Conv1D call signature
        x: Float[Array, "lru_width seq"] = eqx.filter_vmap(
            self.linear_x, in_axes=0, out_axes=1
        )(u)
        x: Float[Array, "seq lru_width "] = rearrange(
            self.conv1d(x), "width seq -> seq width"
        )
        x: Float[Array, "seq lru_width"] = self.lru(x)

        x = x * y
        return eqx.filter_vmap(self.linear_out)(x)


class ResidualBlock(eqx.Module):
    # use_mlp part not implemented as this is not used in the TRecViT architecture
    pre_norm: eqx.nn.RMSNorm
    recurrent_block: RecurrentBlock

    channels: int = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        recurrent_width: int,
        num_heads: int,
        conv1d_kernel_size: int = 4,
        lru_a_init_minval: float = 0.9,
        use_mlp: bool = False,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        self.pre_norm = eqx.nn.RMSNorm(
            shape=(channels,), use_weight=True, use_bias=False, dtype=dtype
        )
        self.recurrent_block = RecurrentBlock(
            channels=channels,
            lru_width=recurrent_width,
            conv1d_kernel_size=conv1d_kernel_size,
            lru_num_heads=num_heads,
            lru_a_init_minval=lru_a_init_minval,
            dtype=dtype,
            key=key,
        )

        self.channels = channels

    def __call__(
        self, u: Float[Array, "seq channels"], *, key: PRNGKeyArray | None = None
    ):
        v = eqx.filter_vmap(self.pre_norm)(u)
        v = self.recurrent_block(v, key=key)

        u = u + v
        # Need to implement optional mlp. Not used in TRecViT
        return u


class TRecViTEncoder(AbstractEncoder):
    num_spatial_dims: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    in_timesteps: ClassVar[None] = None
    embedding_dim: int = eqx.field(static=True)

    tokenizer: Tokenizer
    temporal_layers: list[ResidualBlock]
    spatial_layers: list[TransformerEncoderBlock]
    norm: eqx.nn.LayerNorm

    depth: int = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        grid_size: int | Sequence[int],
        patch_size: int | Sequence[int],
        embedding_dim: int,
        depth: int,
        temporal_block_width: int,
        num_heads: int,
        mlp_hidden_dim: int,
        lru_a_init_minval: float = 0.9,
        dropout=0.0,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, 3)
        self.tokenizer = Tokenizer(
            num_spatial_dims=num_spatial_dims,
            grid_size=grid_size,
            patch_size=patch_size,
            in_dim=in_channels,
            embedding_dim=embedding_dim,
            dtype=dtype,
            key=keys[0],
        )
        self.temporal_layers = [
            ResidualBlock(
                channels=embedding_dim,
                recurrent_width=temporal_block_width,
                num_heads=num_heads,
                lru_a_init_minval=lru_a_init_minval,
                dtype=dtype,
                key=k,
            )
            for k in jax.random.split(keys[1], depth)
        ]
        self.spatial_layers = [
            TransformerEncoderBlock(
                num_heads=num_heads,
                embedding_dim=embedding_dim,
                hidden_dim=mlp_hidden_dim,
                dropout=dropout,
                key=k,
            )
            for k in jax.random.split(keys[2], depth)
        ]
        self.norm = eqx.nn.LayerNorm(shape=(embedding_dim,), dtype=dtype)

        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.depth = depth

    def __call__(
        self,
        u: Float[Array, "time channel *grids"],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Float[Array, " embedding_dim"]:
        keys = maybe_split(key, self.depth)
        u = jax.image.resize(
            u,
            (
                u.shape[0],
                u.shape[1],
                *[
                    s * p
                    for s, p in zip(
                        self.tokenizer.position_encoding.encodings.shape[1:],
                        self.tokenizer.patch_embedding.patch_size,
                    )
                ],
            ),
            method="linear",
        )
        print(u.shape)
        v: Float[Array, "time embedding_dim *grids_patch"] = eqx.filter_vmap(
            lambda x: self.tokenizer(x)
        )(u)
        v = rearrange(v, "t embed ... -> t (...) embed")
        for temporal, spatial, k in zip(
            self.temporal_layers, self.spatial_layers, keys
        ):
            v: Float[Array, "time tokens embedding_dim"] = eqx.filter_vmap(
                temporal, in_axes=1, out_axes=1
            )(v)
            v: Float[Array, "time tokens embedding_dim"] = eqx.filter_vmap(
                partial(spatial, key=k, inference=inference), in_axes=0, out_axes=0
            )(v)
        v = eqx.filter_vmap(self.norm)(rearrange(v, "t tok embed -> (t tok) embed"))
        v = rearrange(v, "(t tok) embed -> t tok embed", t=u.shape[0])
        # Reduce by taking mean over tokens and taking the last time point
        # Like neural CDEs, consider final state of controlled trajectory as the encoded
        # representation
        return jnp.mean(v[-1], axis=0)
