from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class DownsamplingBlock(eqx.Module):
    conv1: eqx.nn.Conv
    conv2: eqx.nn.Conv
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    activation: Callable

    num_spatial_dims: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        groups: int,
        padding_mode: Literal["REFLECT", "CIRCULAR"],
        groups_norm: int,
        hidden_channels: int | None = None,
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        key1, key2 = jax.random.split(key)

        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.conv1 = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            groups=1,
            padding_mode=padding_mode,
            dtype=dtype,
            key=key1,
        )
        self.norm1 = eqx.nn.GroupNorm(
            groups=groups_norm, channels=hidden_channels, channelwise_affine=False
        )
        self.conv2 = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            padding_mode=padding_mode,
            dtype=dtype,
            key=key2,
        )
        self.norm2 = eqx.nn.GroupNorm(
            groups=groups_norm, channels=out_channels, channelwise_affine=False
        )
        self.activation = activation

        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __call__(
        self,
        u: Float[Array, " in_channels *dims"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " out_channels  *dims//2"]:
        u = self.activation(self.norm1(self.conv1(u)))
        u = self.activation(self.norm2(self.conv2(u)))
        return u


class UpSamplingBlock(eqx.Module):
    conv_transpose: eqx.nn.ConvTranspose
    conv: eqx.nn.Conv
    norm: eqx.nn.GroupNorm
    activation: Callable

    num_spatial_dims: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        groups: int,
        padding_mode: Literal["REFLECT", "CIRCULAR"],
        groups_norm: int,
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        key1, key2 = jax.random.split(key)
        hidden_channels = in_channels // 2  # For concatenation; see self.__call__
        self.conv_transpose = eqx.nn.ConvTranspose(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            groups=1,
            dtype=dtype,
            key=key1,
        )
        self.norm = eqx.nn.GroupNorm(
            groups=groups_norm, channels=hidden_channels, channelwise_affine=False
        )
        self.conv2 = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            padding_mode=padding_mode,
            dtype=dtype,
            key=key2,
        )
        self.activation = activation

        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __call__(
        self,
        u: Float[Array, " in_channels *dims//2"],
        v: Float[Array, " in_channels//2 *dims"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " out_channels  *dims//2"]:
        u: Float[Array, " in_channels//2 *dims"] = self.conv_transpose(u)
        u = jnp.concatenate([u, v], axis=0)
        u = self.activation(self.norm(u))
        return u


class OperatorNetwork(eqx.Module):
    downsampling_blocks: list[DownsamplingBlock]
    upsampling_blocks: list[UpSamplingBlock]

    def __init__(
        self,
        num_spatial_dims: int,
        channels: int,
        hidden_channels_base: int,
        groups_norm: int,
        boundary_condition: str,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, 6)

        padding_mode = "CIRCULAR" if boundary_condition == "periodic" else "REFLECT"

        self.input_block = eqx.nn.Sequential([])

        channel_multiplier = (1, 2, 4, 8)

        keys_down = jax.random.split(keys[1], len(channel_multiplier))
        self.downsampling_blocks = [
            DownsamplingBlock(
                num_spatial_dims=num_spatial_dims,
                in_channels=hidden_channels_base * i,
                out_channels=hidden_channels_base * i * 2,
                groups=hidden_channels_base * i * 2,
                padding_mode=padding_mode,
                groups_norm=groups_norm,
                dtype=dtype,
                key=k,
            )
            for i, k in zip(channel_multiplier, keys_down)
        ]

        keys_up = jax.random.split(keys[2], len(channel_multiplier))
        self.upsampling_blocks = [
            UpSamplingBlock(
                num_spatial_dims=num_spatial_dims,
                in_channels=hidden_channels_base * i * 2,
                out_channels=hidden_channels_base * i,
                groups=hidden_channels_base * i,
                padding_mode=padding_mode,
                groups_norm=groups_norm,
                dtype=dtype,
                key=k,
            )
            for i, k in zip(channel_multiplier[::-1], keys_up)
        ]

        keys_bot = jax.random.split(keys[3], 2)
        self.bottleneck = eqx.nn.Sequential([])

        keys_out = jax.random.split(keys[4], 2)
        self.output_block = eqx.nn.Sequential([])

    def __call__(
        self, u: Float[Array, " channels *dims"], *, key: PRNGKeyArray | None = None
    ) -> Float[Array, " channels *dims"]:
        # Add mask if necessary

        u: Float[Array, " hidden_channels_base *dims"] = self.input_block(u)

        u_downs = []
        for downsample in self.downsampling_blocks:
            u_downs.append(u)
            u = downsample(u)

        u = u + self.bottleneck(u)

        for upsample, v in zip(self.upsampling_blocks, u_downs[::-1]):
            u = upsample(u, v)

        return self.output_block(u)
