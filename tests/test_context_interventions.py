"""Checks of the methods that split a step of `HyperFluxFNOLocal`, and of the rollouts
and context interventions in `context_flux_no.evaluation` that build on them.

The split methods and the refresh rollout are compared with `__call__` and
`rollout`, which they must reproduce, and with a loop that is written out step by
step. The interventions are compared with each other on inputs for which two of them
read the same frames.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from context_flux_no.evaluation import (
    cubic_wave_speed_distance,
    evaluate_context_interventions,
    evaluate_refresh_rollouts,
    evaluate_segments,
    INTERVENTIONS,
    rollout_with_flux_models,
    rollout_with_refresh,
)
from context_flux_no.models.multiphysics import HyperFluxFNOLocal
from jaxtyping import Array, Float


# Agreement of two float32 evaluations of the same sequence of operations in another
# grouping, for order-one states and at most 7 steps.
SAME_OPERATIONS_ATOL = 1e-5
# Agreement of the trapezoidal rule on 129 states with a closed-form integral of a
# polynomial of degree 4 on an interval of length 1. The error of the rule is at most
# (1 / 128)**2 / 12 times the largest second derivative of the integrand, 108 here.
QUADRATURE_ATOL = 1e-3

NUM_CELLS = 16
NUM_FRAMES = 14
CONTEXT_LENGTH = 5
ROLLOUT_STEPS = 7
MODEL_ARGS = (0.005, 1 / NUM_CELLS)


@pytest.fixture(scope="module")
def model() -> HyperFluxFNOLocal:
    """A small model with the default initialization of the hypernetwork, so that the
    generated flux networks depend on the context."""
    return HyperFluxFNOLocal(
        num_spatial_dims=1,
        in_channels=1,
        in_timesteps=None,
        embedding_dim=16,
        encoder_type="TRecViT",
        encoder_kwargs={
            "grid_size": [NUM_CELLS],
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
        key=jax.random.key(0),
    )


@pytest.fixture(scope="module")
def trajectories() -> Float[np.ndarray, " 4 frames 1 x"]:
    """Four trajectories of smooth waves with different shapes and speeds."""
    x = np.arange(NUM_CELLS) / NUM_CELLS
    t = np.arange(NUM_FRAMES)[:, None] * 0.02
    waves = [
        np.sin(2 * np.pi * (x - t)),
        0.5 * np.cos(4 * np.pi * (x + 2 * t)),
        np.sin(2 * np.pi * (x - 3 * t)) + 0.3,
        0.8 * np.sin(6 * np.pi * (x + t)),
    ]
    return np.stack(waves)[:, :, None, :].astype(np.float32)


def test_split_methods_reproduce_call(model, trajectories):
    context = jnp.asarray(trajectories[0, :CONTEXT_LENGTH])
    expected = model(context, MODEL_ARGS)[0]
    flux_models = model.generate_flux_models(context)
    assert np.array_equal(model.step(flux_models, context[-1], MODEL_ARGS), expected)


def test_generated_flux_models_depend_on_the_context(model, trajectories):
    # Without this dependence the comparisons of the interventions below would hold
    # for a model that ignores its context.
    state = jnp.asarray(trajectories[0, CONTEXT_LENGTH - 1])
    steps = [
        model.step(
            model.generate_flux_models(jnp.asarray(u[:CONTEXT_LENGTH])),
            state,
            MODEL_ARGS,
        )
        for u in trajectories[:2]
    ]
    assert float(jnp.abs(steps[0] - steps[1]).max()) > 1e-4


def test_refresh_at_every_step_is_the_rollout_of_the_model(model, trajectories):
    context = jnp.asarray(trajectories[2, :CONTEXT_LENGTH])
    expected = model.rollout(context, MODEL_ARGS, num_steps=ROLLOUT_STEPS)[0]
    u_pred = rollout_with_refresh(model, context, MODEL_ARGS, ROLLOUT_STEPS, 1)
    assert np.allclose(u_pred, expected, atol=SAME_OPERATIONS_ATOL)


@pytest.mark.parametrize("refresh_every", [2, 3, ROLLOUT_STEPS, ROLLOUT_STEPS + 4])
def test_refresh_matches_a_loop_written_out(model, trajectories, refresh_every):
    context = jnp.asarray(trajectories[1, :CONTEXT_LENGTH])
    window, flux_models, expected = context, None, []
    for step in range(ROLLOUT_STEPS):
        if step % refresh_every == 0:
            flux_models = model.generate_flux_models(window)
        u_next = model.step(flux_models, window[-1], MODEL_ARGS)
        expected.append(u_next)
        window = jnp.concatenate((window[1:], u_next[None]))
    u_pred = rollout_with_refresh(
        model, context, MODEL_ARGS, ROLLOUT_STEPS, refresh_every
    )
    assert u_pred.shape == (ROLLOUT_STEPS, 1, NUM_CELLS)
    assert np.allclose(u_pred, jnp.stack(expected), atol=SAME_OPERATIONS_ATOL)


def test_refresh_rejects_an_interval_below_one(model, trajectories):
    with pytest.raises(ValueError, match="refresh_every"):
        rollout_with_refresh(
            model, jnp.asarray(trajectories[0, :CONTEXT_LENGTH]), MODEL_ARGS, 4, 0
        )


def test_refresh_errors_at_one_step_are_the_segment_errors(model, trajectories):
    window_starts = (0, 2)
    kwargs = dict(context_length=CONTEXT_LENGTH, rollout_steps=ROLLOUT_STEPS)
    refreshed = evaluate_refresh_rollouts(
        model,
        trajectories,
        window_starts,
        MODEL_ARGS,
        refresh_every=(1, ROLLOUT_STEPS),
        batch_size=2,
        **kwargs,
    )
    segments = evaluate_segments(
        model, trajectories, window_starts, MODEL_ARGS, batch_size=2, **kwargs
    )
    assert refreshed.l2_rollout.shape == (len(trajectories), len(window_starts), 2)
    assert np.allclose(
        refreshed.l2_rollout[..., 0], segments.l2_rollout, atol=SAME_OPERATIONS_ATOL
    )
    assert np.allclose(
        refreshed.l_inf_rollout[..., 0],
        segments.l_inf_rollout,
        atol=SAME_OPERATIONS_ATOL,
    )
    # The last frame alone has another error than the whole rollout.
    assert not np.allclose(refreshed.l2_final, refreshed.l2_rollout)


def _interventions(model, trajectories, donors, permutations, window_starts):
    return evaluate_context_interventions(
        model,
        trajectories,
        donors,
        permutations,
        window_starts,
        MODEL_ARGS,
        context_length=CONTEXT_LENGTH,
        rollout_steps=ROLLOUT_STEPS,
        chunk_size=3,
        batch_size=2,
    )


def test_interventions_that_read_the_true_context_agree(model, trajectories):
    window_starts = (0, 2)
    num_trajectories = len(trajectories)
    # Every trajectory is its own first donor, the identity permutation leaves the
    # context in order, and the second donor is another trajectory.
    donors = np.stack(
        (np.arange(num_trajectories), (np.arange(num_trajectories) + 1) % 4), axis=1
    )
    identity = np.broadcast_to(
        np.arange(CONTEXT_LENGTH),
        (num_trajectories, len(window_starts), CONTEXT_LENGTH),
    )
    errors = _interventions(model, trajectories, donors, identity, window_starts)
    true, shuffled, static = (INTERVENTIONS.index(name) for name in INTERVENTIONS)
    own_donor, other_donor = len(INTERVENTIONS), len(INTERVENTIONS) + 1
    for field in ("l2_onestep", "l_inf_onestep", "l2_rollout", "l_inf_rollout"):
        values = getattr(errors, field)
        assert values.shape == (num_trajectories, len(window_starts), 5)
        assert np.allclose(
            values[..., shuffled], values[..., true], atol=SAME_OPERATIONS_ATOL
        )
        assert np.allclose(
            values[..., own_donor], values[..., true], atol=SAME_OPERATIONS_ATOL
        )
    assert not np.allclose(errors.l2_onestep[..., static], errors.l2_onestep[..., true])
    assert not np.allclose(
        errors.l2_onestep[..., other_donor], errors.l2_onestep[..., true]
    )


def test_true_context_gives_the_rollout_with_held_flux_models(model, trajectories):
    window_starts = (1,)
    donors = np.zeros((len(trajectories), 0), dtype=np.int64)
    permutations = np.broadcast_to(
        np.arange(CONTEXT_LENGTH)[::-1], (len(trajectories), 1, CONTEXT_LENGTH)
    )
    errors = _interventions(model, trajectories, donors, permutations, window_starts)
    held = evaluate_refresh_rollouts(
        model,
        trajectories,
        window_starts,
        MODEL_ARGS,
        context_length=CONTEXT_LENGTH,
        rollout_steps=ROLLOUT_STEPS,
        refresh_every=(ROLLOUT_STEPS,),
        batch_size=2,
    )
    true = INTERVENTIONS.index("true")
    assert np.allclose(
        errors.l2_rollout[..., true], held.l2_rollout[..., 0], atol=SAME_OPERATIONS_ATOL
    )
    # A reversed context is another input to the recurrent encoder.
    shuffled = INTERVENTIONS.index("shuffled")
    assert not np.allclose(
        errors.l2_onestep[..., shuffled], errors.l2_onestep[..., true]
    )


def test_a_missing_donor_has_no_errors(model, trajectories):
    donors = np.asarray([[1], [-1], [3], [-1]])
    permutations = np.broadcast_to(
        np.arange(CONTEXT_LENGTH), (len(trajectories), 1, CONTEXT_LENGTH)
    )
    errors = _interventions(model, trajectories, donors, permutations, (0,))
    donor = len(INTERVENTIONS)
    assert np.isnan(errors.l2_onestep[[1, 3], :, donor]).all()
    assert np.isfinite(errors.l2_onestep[[0, 2], :, donor]).all()
    assert np.isfinite(errors.l2_onestep[..., :donor]).all()


def test_rollout_with_flux_models_repeats_one_step(model, trajectories):
    context = jnp.asarray(trajectories[3, :CONTEXT_LENGTH])
    flux_models = model.generate_flux_models(context)
    u_pred = rollout_with_flux_models(model, flux_models, context[-1], MODEL_ARGS, 3)
    expected: Float[Array, " 1 x"] = context[-1]
    for step in range(3):
        expected = model.step(flux_models, expected, MODEL_ARGS)
        assert np.allclose(u_pred[step], expected, atol=SAME_OPERATIONS_ATOL)


def test_cubic_wave_speed_distance_of_closed_forms():
    zero, one = np.asarray(0.0), np.asarray(1.0)
    # f' = 1 and g' = 3 differ by the constant 2.
    assert cubic_wave_speed_distance(
        np.asarray([0.0, 0.0, 1.0]), np.asarray([0.0, 0.0, 3.0]), zero, one
    ) == pytest.approx(2.0)
    # f' = 2 u and g' = 4 u differ by f' itself.
    assert cubic_wave_speed_distance(
        np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, 2.0, 0.0]), zero, one
    ) == pytest.approx(1.0)
    # f' = 1 and g' = 1 + 3 u^2 differ by 3 u^2, whose L2 norm on [0, 1] is
    # sqrt(9 / 5).
    assert cubic_wave_speed_distance(
        np.asarray([0.0, 0.0, 1.0]), np.asarray([1.0, 0.0, 1.0]), zero, one
    ) == pytest.approx(np.sqrt(9 / 5), abs=QUADRATURE_ATOL)
    # At the single state u = 2, f' = 1 and g' = 1 + 12.
    assert cubic_wave_speed_distance(
        np.asarray([0.0, 0.0, 1.0]), np.asarray([1.0, 0.0, 1.0]), 2 * one, 2 * one
    ) == pytest.approx(12.0)


def test_cubic_wave_speed_distance_broadcasts_donors_against_windows():
    receiver = np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    donors = np.asarray(
        [[[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]], [[0.0, 2.0, 0.0], [0.0, 1.0, 0.0]]]
    )
    lower, upper = np.zeros((2, 3)), np.ones((2, 3))
    distance = cubic_wave_speed_distance(
        receiver[:, None, None, :],
        donors[:, None, :, :],
        lower[:, :, None],
        upper[:, :, None],
    )
    assert distance.shape == (2, 3, 2)
    assert np.allclose(distance[0], [2.0, 0.0])
    assert np.allclose(distance[1], [1.0, 0.0])
