"""Helpers that several test files share."""

import equinox as eqx
import jax
from context_flux_no.models.multiphysics import ContextAppendedFluxNO, HyperFluxFNOLocal
from jaxtyping import PRNGKeyArray


# The factor on the last trunk layer of the small concatenation model. With it, two
# test contexts change one step by about 2.5e-4, above the tolerance of 1e-5 of the
# comparisons in `test_context_interventions.py`.
CONCATENATION_TRUNK_SCALE = 30.0


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


def make_small_concatenation(
    num_cells: int, key: PRNGKeyArray = jax.random.key(0)
) -> ContextAppendedFluxNO:
    """Return a small 1D scalar concatenation variant (Table 1) with the encoder and
    stencil of `make_small_hfluxno`.

    `ContextConditionedFluxModel` broadcasts the context vector to the shape of the
    lifted stencil features, so `lift_dim` equals `embedding_dim`, as it does in
    `fluxno_appended.yaml`. At its default initialization the step of this model
    changes by about 1e-5 between two contexts, below the tolerance of the comparisons
    that use it, so the last layer of the hypernetwork trunk is scaled by
    `CONCATENATION_TRUNK_SCALE`. `num_cells` must be a multiple of the patch size 4.
    """
    model = ContextAppendedFluxNO(
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
        lift_dim=16,
        stencil_size=(2, 2),
        width_flux=8,
        width_hyper=16,
        depth_hyper=1,
        key=key,
    )
    return eqx.tree_at(
        lambda m: m.hypernetwork_trunk.layers[-1].weight,
        model,
        replace_fn=lambda weight: CONCATENATION_TRUNK_SCALE * weight,
    )
