from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import pack, rearrange, unpack
from equinox._misc import default_floating_dtype
from jaxtyping import Array, Float, PRNGKeyArray

from ....nn.attention import MultiheadAttention
from ....nn.drop import DropPath
from ....nn.misc import apply_along_axis, maybe_split
from ....nn.position_encoding import T5RelativePositionalEncoding


class AttentionBlock1D(eqx.Module):
    """Apply 1D multihead self-attention along the last two dimensions of the input
    tensor. Expects input tensor to be the shape (..., sequence, channels).

    By default, use QK-Norm and T5RelativePositionalEncoding.

    Note that this block is not present in the original DICSO code. Indeed, this is the
    refactored out common logic underlying the TemporalAttention and SpatialAttention
    blocks."""

    attention: MultiheadAttention
    rel_pos_encoding: T5RelativePositionalEncoding

    channels: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        num_heads: int,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, 2)

        self.rel_pos_encoding = T5RelativePositionalEncoding(
            num_encodings=32,
            num_heads=num_heads,
            bidirectional=True,
            max_distance=128,
            dtype=dtype,
            key=keys[0],
        )
        # Equivalent to the Conv3D and LayerNorms used in the original implementation
        self.attention = MultiheadAttention(
            num_heads=num_heads,
            query_size=channels,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            use_qk_norm=True,
            use_output_proj=False,
            dropout_p=0.0,
            dtype=dtype,
            key=keys[1],
        )

        self.channels = channels
        self.num_heads = num_heads

    def check_inputs(self, u: Float[Array, "*dims sequence channels"]) -> None:
        if u.ndim < 2:
            raise ValueError("Input must have at least two dimensions.")
        if u.shape[-1] != self.channels:
            raise ValueError(f"Size of the last axis must match {self.channels=}")

    def __call__(
        self,
        u: Float[Array, "*dims sequence channels"],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Float[Array, "*dims sequence channels"]:
        self.check_inputs(u)

        u, shape_info = pack([u], "* S C")

        rel_pos_enc = self.rel_pos_encoding(u.shape[1], u.shape[1])
        attention = partial(
            self.attention, bias=rel_pos_enc, key=key, inference=inference
        )
        u: Float[Array, "batch sequence channels"] = eqx.filter_vmap(attention)(u, u, u)
        return unpack(u, shape_info, "* S C")[0]


class TemporalAttention(eqx.Module):
    """Apply 1D attention across the first two dimensions of the input tensor, with the
    first dimension being the sequence length, and the second dimension being the
    channels (i.e. features). The operation is batched over the remaining trailing axes.
    """

    attention_1d: AttentionBlock1D
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    output_proj: eqx.nn.Linear
    layer_scale: Float[Array, " {self.channels}"]
    droppath: DropPath

    channels: int = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        num_heads: int,
        num_groups: int | None = None,
        layer_scale_init: float = 1e-6,
        droppath: float = 0.0,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        dtype = default_floating_dtype() if dtype is None else dtype
        keys = jax.random.split(key, 2)
        num_groups = num_heads if num_groups is None else num_groups

        self.norm1 = eqx.nn.GroupNorm(num_groups, channels, dtype=dtype)
        self.norm2 = eqx.nn.GroupNorm(num_groups, channels, dtype=dtype)

        self.attention_1d = AttentionBlock1D(channels, num_heads, dtype, key=keys[0])
        self.output_proj = eqx.nn.Linear(channels, channels, dtype=dtype, key=keys[1])
        self.layer_scale = jnp.full(
            shape=(channels,), fill_value=layer_scale_init, dtype=dtype
        )
        self.droppath = DropPath(droppath)

        self.channels = channels

    def __call__(
        self,
        u: Float[Array, "time channels *spatial_dims"],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Float[Array, "time channels *spatial_dims"]:
        keys = maybe_split(key, 2)

        u = rearrange(u, "S C ... -> ... S C")
        v: Float[Array, "*spatial_dims sequence channels"] = apply_along_axis(
            self.norm1, u, axis=-1
        )
        v = self.attention_1d(v, key=keys[0], inference=inference)
        v = apply_along_axis(lambda x: self.output_proj(self.norm2(x)), v, axis=-1)

        u: Float[Array, "*spatial_dims sequence channels"] = u + self.droppath(
            v * self.layer_scale, key=keys[1], inference=inference
        )
        return rearrange(u, "... S C -> S C ...")


class SpatialAxialAttention(eqx.Module):
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    norm_mlp: eqx.nn.GroupNorm
    attention_per_axis: list[AttentionBlock1D]
    output_proj: eqx.nn.Linear
    mlp: eqx.nn.MLP
    layer_scale_attention: Float[Array, " {self.channels}"]
    layer_scale_mlp: Float[Array, " {self.channels}"]
    droppath: DropPath

    num_spatial_dims: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        channels: int,
        num_heads: int,
        num_groups: int,
        layer_scale_init: float = 1e-6,
        droppath: float = 0.0,
        mlp_hidden_ratio: float = 4.0,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, num_spatial_dims + 2)

        self.norm1 = eqx.nn.GroupNorm(num_groups, channels, dtype=dtype)
        self.norm2 = eqx.nn.GroupNorm(num_groups, channels, dtype=dtype)
        self.norm_mlp = eqx.nn.GroupNorm(num_groups, channels, dtype=dtype)

        self.attention_per_axis = [
            AttentionBlock1D(channels, num_heads, dtype, key=k)
            for k in keys[:num_spatial_dims]
        ]
        self.mlp = eqx.nn.MLP(
            in_size=channels,
            out_size=channels,
            width_size=int(mlp_hidden_ratio * channels),
            depth=1,
            activation=jax.nn.gelu,
            dtype=dtype,
            key=keys[-2],
        )
        self.layer_scale_attention = jnp.full(
            shape=(channels,), fill_value=layer_scale_init, dtype=dtype
        )
        self.layer_scale_mlp = jnp.full(
            shape=(channels,), fill_value=layer_scale_init, dtype=dtype
        )

        self.output_proj = eqx.nn.Linear(channels, channels, dtype=dtype, key=keys[-1])
        self.droppath = DropPath(droppath)

        self.num_spatial_dims = num_spatial_dims
        self.channels = channels

    def __call__(
        self,
        u: Float[Array, "channels patch_x patch_y patch_z"],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ):
        if u.ndim != self.num_spatial_dims + 1:
            raise ValueError(
                f"Number of dimensions of u must equal {self.num_spatial_dims}+1"
            )

        u = rearrange(u, "C ... -> ... C")
        v: Float[Array, "*spatial_dims channels"] = apply_along_axis(
            self.norm1, u, axis=-1
        )

        keys = maybe_split(key, self.num_spatial_dims + 2)

        def apply_axial_attention(
            x: Float[Array, "*spatial_dims channels"],
            axis: int,
            attention: AttentionBlock1D,
            *,
            key_: PRNGKeyArray | None,
        ) -> Float[Array, "*spatial_dims channels"]:
            x = jnp.swapaxes(x, axis, x.ndim - 2)
            x = attention(x, key=key_, inference=inference)
            x = jnp.swapaxes(x, axis, x.ndim - 2)
            return x

        v_out = sum(
            [
                apply_axial_attention(v, i, attn, key_=keys[i])
                for i, attn in enumerate(self.attention_per_axis)
            ]
        )
        v_out: Float[Array, "*spatial_dims channels"] = v_out / self.num_spatial_dims
        v_out = apply_along_axis(
            lambda x: self.output_proj(self.norm2(x)), v_out, axis=-1
        )
        u: Float[Array, "*spatial_dims channels"] = u + self.droppath(
            v_out * self.layer_scale_attention, key=keys[-2], inference=inference
        )

        v_mlp = apply_along_axis(lambda x: self.norm_mlp(self.mlp(x)), u, axis=-1)
        u: Float[Array, "*spatial_dims channels"] = u + self.droppath(
            v_mlp * self.layer_scale_mlp, key=keys[-1], inference=inference
        )
        return rearrange(u, "... C -> C ...")
