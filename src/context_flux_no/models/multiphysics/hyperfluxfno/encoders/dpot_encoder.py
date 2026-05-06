from collections.abc import Callable

import equinox as eqx
import jax
from einops import reduce
from jaxtyping import Array, Float, PRNGKeyArray

from context_flux_no.models.multiphysics.dpot import DPOTBlock, TimeAggregator
from context_flux_no.nn import PatchEmbedding
from context_flux_no.nn.misc import to_ntuple
from context_flux_no.nn.position_encoding import LearnedPositionEncoding

from .base import AbstractEncoder


class DPOTEncoder(AbstractEncoder):
    """An encoder implementation based on the TimeAggregator and DPOTBlock modules
    introduced in the DPOT paper. Treats space and time separately and is very fast,
    but only supports fixed timesteps."""

    num_spatial_dims: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    in_timesteps: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)

    patch_embedding: PatchEmbedding
    position_embedding: LearnedPositionEncoding
    time_aggregator: TimeAggregator
    blocks: list[DPOTBlock]

    grid_size: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        in_timesteps: int,
        grid_size: int | tuple[int, ...],
        patch_size: int | tuple[int, ...],
        embedding_dim: int,
        max_frequency_modes: int | tuple[int, ...],
        fno_depth: int,
        num_blocks: int,
        hidden_dim_patch: int,
        hidden_dim_fno: int,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, 4)

        grid_size = to_ntuple(grid_size, num_spatial_dims)
        patch_size = to_ntuple(patch_size, num_spatial_dims)
        self.patch_embedding = PatchEmbedding(
            num_spatial_dims=num_spatial_dims,
            patch_size=patch_size,
            in_dim=in_channels,
            embedding_dim=embedding_dim,
            num_hidden=1,
            hidden_dim=hidden_dim_patch,
            activation=activation,
            dtype=dtype,
            key=keys[0],
        )
        self.position_embedding = LearnedPositionEncoding(
            channels=embedding_dim,
            spatial_dims=self.patch_embedding.output_size(grid_size),
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

        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.in_timesteps = in_timesteps
        self.embedding_dim = embedding_dim
        self.grid_size = grid_size

    def __call__(
        self,
        u: Float[Array, "time channels *grids"],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Float[Array, " embedding_dim"]:
        # Model forward pass is deterministic
        del key, inference

        # TODO: Check if normalization is actually used and implement if necessary
        if u.shape[0] != self.in_timesteps:
            raise ValueError(
                "Input array time dimension does not match self.in_timesteps"
            )
        if u.shape[1] != self.in_channels:
            raise ValueError(
                "Input array channel dimension does not match self.in_channels"
            )
        if u.shape[2:] != self.grid_size:
            raise ValueError(
                "Input array spatial dimensions do not match self.img_size"
            )

        # Patch embedding for the spatial dimensions
        v: Float[Array, "time channels_embed *patches"] = eqx.filter_vmap(
            self.patch_embedding
        )(u)

        # Apply positional embedding
        v: Float[Array, "time channels_embed *patches"] = eqx.filter_vmap(
            self.position_embedding
        )(v)

        # Time aggregation layer
        v: Float[Array, " channels_embed *patches"] = self.time_aggregator(v)

        for dpot_block in self.blocks:
            v: Float[Array, " channels_embed *patches"] = dpot_block(v)

        return reduce(v, "C ... -> C", "mean")
