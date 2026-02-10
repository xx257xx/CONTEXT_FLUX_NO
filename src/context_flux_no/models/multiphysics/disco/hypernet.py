from typing import Literal

import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from ....nn.misc import atleast_nd
from .attention_blocks import SpatialAxialAttention, TemporalAttention


class SpaceTimeBlock(eqx.Module):
    temporal_block: TemporalAttention
    spatial_block: SpatialAxialAttention

    channels: int = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        num_heads: int,
        droppath: float,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        key_t, key_s = jax.random.split(key, 2)
        self.temporal_block = TemporalAttention(
            channels=channels,
            num_heads=num_heads,
            num_groups=num_heads,
            layer_scale_init=1e-6,
            droppath=droppath,
            dtype=dtype,
            key=key_t,
        )
        self.spatial_block = SpatialAxialAttention(
            num_spatial_dims=3,
            channels=channels,
            num_heads=num_heads,
            num_groups=num_heads * 2,
            layer_scale_init=1e-6,
            droppath=droppath,
            dtype=dtype,
            key=key_s,
        )

        self.channels = channels

    def __call__(
        self,
        u: Float[Array, "time channels patch_x patch_y patch_z"],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Float[Array, "time channels patch_x patch_y patch_z"]:
        if key is not None:
            keys = jax.random.split(key, 2)
        else:
            keys = (None, None)

        u = self.temporal_block(u, key=keys[0], inference=inference)
        u = eqx.filter_vmap(
            lambda x: self.spatial_block(x, key=keys[1], inference=inference)
        )(u)
        return u


class HyperNetwork(eqx.Module):
    adapter: eqx.nn.Linear
    encoders: dict[int, eqx.nn.Sequential]
    blocks: list[SpaceTimeBlock]

    in_channels: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)
    padding_mode: Literal["ZEROS", "REPLICATE", "REFLECT", "CIRCULAR"] = eqx.field(
        static=True
    )

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        embedding_dim: int,
        num_blocks: int,
        droppath: float,
        num_heads: int,
        padding_mode: Literal["ZEROS", "REPLICATE", "REFLECT", "CIRCULAR"] = "REFLECT",
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.padding_mode = padding_mode

        key_a, key_e, key_b = jax.random.split(key, 3)

        # For now, do not care about variable in_channels
        self.adapter = eqx.nn.Linear(
            in_features=in_channels,
            out_features=self.embedding_dim // 4,
            dtype=dtype,
            key=key_a,
        )
        # Encoders for 1D, 2D, and 3D
        self.encoders = {
            dim: self.make_encoder(dim, k)
            for dim, k in zip((1, 2, 3), jax.random.split(key_e, 3))
        }

        per_block_droppaths = np.linspace(0, droppath, num_blocks)
        self.blocks = [
            SpaceTimeBlock(
                channels=embedding_dim,
                num_heads=num_heads,
                droppath=dp,
                dtype=dtype,
                key=k,
            )
            for dp, k in zip(per_block_droppaths, jax.random.split(key_b, num_blocks))
        ]

    def make_encoder(
        self, num_spatial_dims: int, key: PRNGKeyArray
    ) -> eqx.nn.Sequential:
        in_channels, out_channels = self.embedding_dim // 4, self.embedding_dim
        kernel_size = self.patch_size // 4

        keys = jax.random.split(key, 3)
        encoder = eqx.nn.Sequential(
            [
                eqx.nn.Conv(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=kernel_size,
                    padding=0,
                    padding_mode=self.padding_mode,
                    key=keys[0],
                ),
                eqx.nn.GroupNorm(groups=1, channels=in_channels),
                eqx.nn.Lambda(jax.nn.gelu),
                eqx.nn.Conv(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    padding_mode=self.padding_mode,
                    key=keys[1],
                ),
                eqx.nn.GroupNorm(groups=1, channels=in_channels),
                eqx.nn.Lambda(jax.nn.gelu),
                eqx.nn.Conv(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    padding_mode=self.padding_mode,
                    key=keys[2],
                ),
                eqx.nn.GroupNorm(groups=1, channels=out_channels),
            ]
        )
        return encoder

    def __call__(
        self,
        u: Float[Array, "time in_channels *grids"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "time embedding_dim patch_x patch_y patch_z"]:
        num_spatial_dims = u.ndim - 2

        # Add dimension manipulation and standardization here

        u: Float[Array, "time embedding_dim//4 *grids"] = self.adapter(u)

        v: Float[Array, "time embedding_dim *patches"] = eqx.filter_vmap(
            self.encoders[num_spatial_dims]
        )(u)
        v: Float[Array, "time embedding_dim patch_x patch_y patch_z"] = atleast_nd(
            v, n=5
        )

        for block in self.blocks:
            v = block(v)
        return v
