from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import reduce
from equinox._misc import default_floating_dtype
from jaxtyping import Array, Float, PRNGKeyArray

from ...nn.embedding import PatchEmbedding
from ...nn.misc import to_ntuple
from ...nn.operators import AdaptiveFourier
from ...nn.operators.fourier_utils import append_grid_channels
from ...nn.position_encoding import LearnedPositionEncoding


class TimeAggregator(eqx.Module):
    fourier_freqs: Float[Array, " channels"]
    weights: Float[Array, "timesteps channels channels"]

    timesteps: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)

    def __init__(self, timesteps: int, channels: int, dtype=None, *, key: PRNGKeyArray):
        dtype = default_floating_dtype() if dtype is None else dtype
        self.fourier_freqs = jnp.logspace(-10, 10, channels, base=2, dtype=dtype)

        scale = 1 / (timesteps * channels**0.5)
        self.weights = scale * jax.random.normal(
            key, shape=(timesteps, channels, channels), dtype=dtype
        )
        self.timesteps = timesteps
        self.channels = channels

    def __call__(
        self, u: Float[Array, "time channels *grids"]
    ) -> Float[Array, " channels *grids"]:
        with jax.ensure_compile_time_eval():
            t = jnp.linspace(0, 1, u.shape[0])
        fourier_features: Float[Array, "time channels"] = jnp.cos(
            jnp.outer(t, self.fourier_freqs)
        )
        return jnp.einsum("tij,ti...,ti->j...", self.weights, u, fourier_features)


class DPOTBlock(eqx.Module):
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    spatial_mixing: AdaptiveFourier
    channel_mixing: eqx.nn.Sequential

    num_spatial_dims: int = eqx.field(static=True)
    double_skip: bool = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        channels: int,
        max_frequency_modes: int | tuple[int, ...],
        channels_hidden: int,
        groups: int = 8,
        num_blocks: int = 4,
        double_skip: bool = True,
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        self.norm1 = eqx.nn.GroupNorm(groups, channels, dtype=dtype)
        self.norm2 = eqx.nn.GroupNorm(groups, channels, dtype=dtype)

        keys = jax.random.split(key, 3)
        self.spatial_mixing = AdaptiveFourier(
            num_spatial_dims=num_spatial_dims,
            in_channels=channels,
            out_channels=channels,
            max_frequency_modes=max_frequency_modes,
            hidden_channels=channels,  # Consistent with original DPOT parameters
            num_blocks=num_blocks,
            activation=activation,
            dtype=dtype,
            key=keys[0],
        )
        self.channel_mixing = eqx.nn.Sequential(
            [
                eqx.nn.Conv(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=channels,
                    out_channels=channels_hidden,
                    kernel_size=1,
                    stride=1,
                    dtype=dtype,
                    key=keys[1],
                ),
                eqx.nn.Lambda(jax.nn.gelu),
                eqx.nn.Conv(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=channels_hidden,
                    out_channels=channels,
                    kernel_size=1,
                    stride=1,
                    dtype=dtype,
                    key=keys[2],
                ),
            ]
        )
        self.num_spatial_dims = num_spatial_dims
        self.double_skip = double_skip

    def __call__(
        self,
        x: Float[Array, " channels *patches"],
    ) -> Float[Array, " channels *patches"]:
        y: Float[Array, " channels *patches"] = self.spatial_mixing(self.norm1(x))

        if self.double_skip:
            y = y + x
            x = y

        y: Float[Array, " channels *patches"] = self.channel_mixing(self.norm2(y))
        y = y + x
        return y


class DPOT(eqx.Module):
    """JAX implementation of the DPOT model presented in [1].

    [1] Z. Hao et al. DPOT: Auto-Regressive Denoising Operator Transformer for
    Large-Scale PDE Pre-Training. ICML (2024)."""

    patch_embedding: PatchEmbedding
    position_embedding: LearnedPositionEncoding
    time_aggregator: TimeAggregator
    blocks: list[DPOTBlock]
    output_head: eqx.nn.Sequential
    classification_head: eqx.nn.MLP

    num_spatial_dims: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    in_timesteps: int = eqx.field(static=True)
    img_size: tuple[int, ...] = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        in_timesteps: int,
        img_size: int | tuple[int],
        patch_size: int | tuple[int],
        embedding_dim: int,
        max_frequency_modes: int | tuple[int],
        fno_depth: int,
        num_blocks: int,
        num_classes: int,
        hidden_dim_patch: int,
        hidden_dim_fno: int,
        hidden_dim_output: int,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, 6)

        img_size = to_ntuple(img_size, num_spatial_dims)
        patch_size = to_ntuple(patch_size, num_spatial_dims)
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
        self.position_embedding = LearnedPositionEncoding(
            channels=embedding_dim,
            spatial_dims=self.patch_embedding.output_size(img_size),
            init_scale=0.02,
            dtype=dtype,
            key=keys[1],
        )
        self.time_aggregator = TimeAggregator(
            timesteps=in_timesteps, channels=embedding_dim, dtype=dtype, key=keys[2]
        )
        subkeys = jax.random.split(keys[3], fno_depth)
        self.blocks = [
            DPOTBlock(
                num_spatial_dims=num_spatial_dims,
                channels=embedding_dim,
                max_frequency_modes=max_frequency_modes,
                channels_hidden=hidden_dim_fno,
                num_blocks=num_blocks,
                double_skip=False,
                activation=activation,
                dtype=dtype,
                key=k,
            )
            for k in subkeys
        ]

        subkeys2 = jax.random.split(keys[4], 3)
        self.output_head = eqx.nn.Sequential(
            [
                eqx.nn.ConvTranspose(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=embedding_dim,
                    out_channels=hidden_dim_output,
                    kernel_size=patch_size,
                    stride=patch_size,
                    dtype=dtype,
                    key=subkeys2[0],
                ),
                eqx.nn.Lambda(activation),
                eqx.nn.Conv(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=hidden_dim_output,
                    out_channels=hidden_dim_output,
                    kernel_size=1,
                    dtype=dtype,
                    key=subkeys2[1],
                ),
                eqx.nn.Lambda(activation),
                eqx.nn.Conv(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=hidden_dim_output,
                    out_channels=out_channels,
                    kernel_size=1,
                    dtype=dtype,
                    key=subkeys2[2],
                ),
            ]
        )
        self.classification_head = eqx.nn.MLP(
            in_size=embedding_dim,
            out_size=num_classes,
            width_size=embedding_dim,
            depth=2,
            activation=activation,
            dtype=dtype,
            key=keys[5],
        )

        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_timesteps = in_timesteps
        self.img_size = img_size
        self.num_classes = num_classes

    def __call__(
        self,
        u: Float[Array, "time channels grid_x"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "channels grid_x"], Float[Array, " num_classes"]]:
        del key
        # TODO: Check if normalization is actually used and implement if necessary
        if u.shape[0] != self.in_timesteps:
            raise ValueError(
                "Input array time dimension does not match self.in_timesteps"
            )
        if u.shape[1] != self.in_channels:
            raise ValueError(
                "Input array channel dimension does not match self.in_channels"
            )
        if u.shape[2:] != self.img_size:
            raise ValueError(
                "Input array spatial dimensions do not match self.img_size"
            )

        u: Float[Array, "time channels+num_spatial_dims grid_x"] = jax.vmap(
            append_grid_channels
        )(u)

        # Patch embedding for the spatial dimensions
        v: Float[Array, "time channels_embed patch_x"] = eqx.filter_vmap(
            self.patch_embedding
        )(u)

        # Apply positional embedding
        v: Float[Array, "time channels_embed patch_x"] = eqx.filter_vmap(
            self.position_embedding
        )(v)

        # Time aggregation layer
        v: Float[Array, "channels_embed patch_x"] = self.time_aggregator(v)

        for dpot_block in self.blocks:
            v: Float[Array, "channels_embed patch_x"] = dpot_block(v)

        u_next = self.output_head(v)

        cls_token: Float[Array, " channels_embed"] = reduce(v, "c ... -> c", "mean")
        cls_pred: Float[Array, " num_classes"] = self.classification_head(cls_token)
        return u_next, cls_pred
