from collections.abc import Callable

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

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

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        in_timesteps: int,
        img_size: int | tuple[int],
        patch_size: int | tuple[int],
        embedding_dim: int,
        max_frequency_modes: int | tuple[int],
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
