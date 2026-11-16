"""Helpers that several test files share."""

import jax
from context_flux_no.models.multiphysics import HyperFluxFNOLocal
from jaxtyping import PRNGKeyArray


def make_small_hfluxno(
    num_cells: int, key: PRNGKeyArray = jax.random.key(0)
) -> HyperFluxFNOLocal:
    """Return a small 1D scalar HFluxNO whose stencil reaches 2 cells to each side.

    The hypernetwork has its default initialization, so the generated flux networks
    depend on the context. `num_cells` must be a multiple of the patch size 4.
    """
    return HyperFluxFNOLocal(
        num_spatial_dims=1,
        in_channels=1,
        in_timesteps=None,
        embedding_dim=16,
        encoder_type="TRecViT",
        encoder_kwargs={
            "grid_size": [num_cells],
            "patch_size": [4],
            "depth": 1,
            "temporal_block_width": 16,
            "num_heads": 2,
            "mlp_hidden_dim": 16,
        },
        depth=1,
        lift_dim=8,
        stencil_size=(2, 2),
        width_flux=8,
        width_hyper=16,
        blocks_hyper=2,
        key=key,
    )
