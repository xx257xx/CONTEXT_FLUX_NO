from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from context_flux_no.nn.convtranspose import ConvTranspose


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
    ) -> Float[Array, " out_channels  *dims_half"]:
        u = self.activation(self.norm1(self.conv1(u)))
        u = self.activation(self.norm2(self.conv2(u)))
        return u


class UpSamplingBlock(eqx.Module):
    conv_transpose: ConvTranspose
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
        self.conv_transpose = ConvTranspose(
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
            groups=groups_norm, channels=out_channels, channelwise_affine=False
        )
        self.conv = eqx.nn.Conv(
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
        u: Float[Array, " in_channels *dims_half"],
        v: Float[Array, " in_channels//2 *dims"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " out_channels  *dims_half"]:
        # Use output padding to support spatial dimensions that are not multiples of 16
        output_padding = tuple(d_v % d_u for d_v, d_u in zip(v.shape[1:], u.shape[1:]))
        u: Float[Array, " in_channels//2 *dims"] = self.conv_transpose(
            u, output_padding
        )
        u = jnp.concatenate([u, v], axis=0)
        u = self.activation(self.norm(self.conv(u)))
        return u


class OperatorNetwork(eqx.Module):
    """Operator network for the DISCO model.

    Note that unlike the original implementation, which requires the spatial dimensions
    to be multiples of 16, this implementation supports arbitrary spatial dimensions as
    long as they are greater than 15 (d//2**4=0 for d<16).
    This is accomplished using output padding in the forward pass of UpSamplingBlock."""

    input_block: eqx.nn.Sequential
    downsampling_blocks: list[DownsamplingBlock]
    bottleneck: eqx.nn.Sequential
    upsampling_blocks: list[UpSamplingBlock]
    output_block: eqx.nn.Sequential

    num_spatial_dims: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    boundary_condition: str = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        channels: int,
        hidden_channels_base: int,
        groups_norm: int,
        boundary_condition: str,
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, 5)

        padding_mode = "CIRCULAR" if boundary_condition == "periodic" else "REFLECT"
        add_boundary_mask = False if boundary_condition == "periodic" else True

        conv_in_out_kwargs = {
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "padding_mode": padding_mode,
            "groups": 1,
            "dtype": dtype,
        }

        keys_in = jax.random.split(keys[0], 2)
        self.input_block = eqx.nn.Sequential(
            [
                eqx.nn.Conv(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=channels + 1 if add_boundary_mask else channels,
                    out_channels=hidden_channels_base,
                    **conv_in_out_kwargs,
                    key=keys_in[0],
                ),
                eqx.nn.Conv(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=hidden_channels_base,
                    out_channels=hidden_channels_base,
                    **conv_in_out_kwargs,
                    key=keys_in[1],
                ),
                eqx.nn.GroupNorm(
                    groups=groups_norm,
                    channels=hidden_channels_base,
                    channelwise_affine=False,
                ),
                eqx.nn.Lambda(activation),
            ]
        )

        channel_multipliers = (1, 2, 4, 8)

        keys_down = jax.random.split(keys[1], len(channel_multipliers))
        self.downsampling_blocks = [
            DownsamplingBlock(
                num_spatial_dims=num_spatial_dims,
                in_channels=hidden_channels_base * i,
                out_channels=hidden_channels_base * i * 2,
                groups=hidden_channels_base * i * 2,
                padding_mode=padding_mode,
                groups_norm=groups_norm,
                activation=activation,
                dtype=dtype,
                key=k,
            )
            for i, k in zip(channel_multipliers, keys_down)
        ]

        keys_up = jax.random.split(keys[2], len(channel_multipliers))
        self.upsampling_blocks = [
            UpSamplingBlock(
                num_spatial_dims=num_spatial_dims,
                in_channels=hidden_channels_base * i * 2,
                out_channels=hidden_channels_base * i,
                groups=hidden_channels_base * i,
                padding_mode=padding_mode,
                groups_norm=groups_norm,
                activation=activation,
                dtype=dtype,
                key=k,
            )
            for i, k in zip(channel_multipliers[::-1], keys_up)
        ]

        keys_bot = jax.random.split(keys[3], 2)
        conv_bottle_kwargs = {
            "kernel_size": 1,
            "stride": 1,
            "padding": 0,
            "padding_mode": padding_mode,
            "groups": 1,
            "dtype": dtype,
        }
        self.bottleneck = eqx.nn.Sequential(
            [
                eqx.nn.Conv(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=hidden_channels_base * channel_multipliers[-1] * 2,
                    out_channels=hidden_channels_base * channel_multipliers[-1] * 4,
                    **conv_bottle_kwargs,
                    key=keys_bot[0],
                ),
                eqx.nn.GroupNorm(
                    groups=groups_norm,
                    channels=hidden_channels_base * channel_multipliers[-1] * 4,
                    channelwise_affine=False,
                ),
                eqx.nn.Lambda(activation),
                eqx.nn.Conv(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=hidden_channels_base * channel_multipliers[-1] * 4,
                    out_channels=hidden_channels_base * channel_multipliers[-1] * 2,
                    **conv_bottle_kwargs,
                    key=keys_bot[0],
                ),
            ]
        )

        keys_out = jax.random.split(keys[4], 2)
        self.output_block = eqx.nn.Sequential(
            [
                eqx.nn.Conv(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=hidden_channels_base,
                    out_channels=hidden_channels_base,
                    **conv_in_out_kwargs,
                    key=keys_out[0],
                ),
                eqx.nn.GroupNorm(
                    groups=groups_norm,
                    channels=hidden_channels_base,
                    channelwise_affine=False,
                ),
                eqx.nn.Lambda(activation),
                eqx.nn.Conv(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=hidden_channels_base,
                    out_channels=channels,
                    **conv_in_out_kwargs,
                    key=keys_out[1],
                ),
            ]
        )
        self.num_spatial_dims = num_spatial_dims
        self.channels = channels
        self.boundary_condition = boundary_condition

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
