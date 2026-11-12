"""Checks of the segment evaluation in `context_flux_no.evaluation`.

The drivers are compared with closed forms for a traveling sine wave, and the grouping
functions with cases that are listed by hand.
"""

from typing import Any

import numpy as np
import pytest
from context_flux_no.evaluation import (
    evaluate_segment_baselines,
    evaluate_segments,
    match_trajectories,
    mean_by_bin,
)
from context_flux_no.models.multiphysics.abstract import AbstractMultiphysicsOperator
from jaxtyping import Array, Float, PRNGKeyArray


# Agreement of a closed form with its float32 evaluation, for sums over at most
# 20 frames of 64 cells of order-one values.
ATOL = 1e-4

NUM_CELLS = 64
NUM_FRAMES = 12
CONTEXT_LENGTH = 3
ROLLOUT_STEPS = 4
# The wave moves by this fraction of its period per frame.
SHIFT_PER_FRAME = 1 / 32


class HoldLastFrame(AbstractMultiphysicsOperator):
    """A model that predicts the last context frame, which is the identity baseline."""

    num_spatial_dims: int = 1

    def __call__(
        self,
        u: Float[Array, " time channels *grids"],
        args: Any,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> tuple[Float[Array, " channels *grids"], None]:
        return u[-1], None


def _traveling_sines() -> Float[np.ndarray, " 2 frames 1 x"]:
    """Two trajectories of a sine wave that moves at a constant speed. The second has
    twice the amplitude, which no relative error depends on."""
    x = np.arange(NUM_CELLS) / NUM_CELLS
    frames = np.arange(NUM_FRAMES)[:, None] * SHIFT_PER_FRAME
    wave = np.sin(2 * np.pi * (x - frames))[:, None, :]
    return np.stack((wave, 2 * wave)).astype(np.float32)


def _identity_errors() -> tuple[float, float]:
    """Single-step and rollout l2 error of the identity baseline on the sine wave.

    A sine shifted by the phase phi differs from itself by 2 |sin(phi / 2)| in relative
    l2. Step n of the rollout has the phase 2 pi n SHIFT_PER_FRAME, and the rollout
    error is the root mean square over the steps, since every frame has the same norm.
    """
    per_step = [
        2 * abs(np.sin(np.pi * n * SHIFT_PER_FRAME))
        for n in range(1, ROLLOUT_STEPS + 1)
    ]
    return per_step[0], float(np.sqrt(np.mean(np.square(per_step))))


def test_baselines_on_a_traveling_sine():
    window_starts = (0, 2, 5)
    baselines = evaluate_segment_baselines(
        _traveling_sines(),
        window_starts,
        context_length=CONTEXT_LENGTH,
        rollout_steps=ROLLOUT_STEPS,
    )
    onestep, rollout = _identity_errors()
    assert baselines.identity.l2_onestep.shape == (2, len(window_starts))
    assert np.allclose(baselines.identity.l2_onestep, onestep, atol=ATOL)
    assert np.allclose(baselines.identity.l2_rollout, rollout, atol=ATOL)
    # The cell mean of a sine is zero, so the spatial-mean baseline predicts zero.
    assert np.allclose(baselines.spatial_mean.l2_onestep, 1.0, atol=ATOL)
    assert np.allclose(baselines.spatial_mean.l2_rollout, 1.0, atol=ATOL)
    assert np.allclose(baselines.spatial_mean.l_inf_rollout, 1.0, atol=ATOL)


def test_a_model_that_holds_the_last_frame_has_the_identity_errors():
    window_starts = (0, 1, 5)
    errors = evaluate_segments(
        HoldLastFrame(),
        _traveling_sines(),
        window_starts,
        (0.005, 1 / NUM_CELLS),
        context_length=CONTEXT_LENGTH,
        rollout_steps=ROLLOUT_STEPS,
        batch_size=1,
    )
    onestep, rollout = _identity_errors()
    assert errors.l2_rollout.shape == (2, len(window_starts))
    assert np.allclose(errors.l2_onestep, onestep, atol=ATOL)
    assert np.allclose(errors.l2_rollout, rollout, atol=ATOL)


def test_each_window_is_scored_against_its_own_frames():
    # Frame n holds the constant n + 1, so the identity baseline of a segment that
    # starts at frame s predicts s + CONTEXT_LENGTH and its single-step error is
    # 1 / (s + CONTEXT_LENGTH + 1). The error identifies the window.
    frames = np.arange(1, NUM_FRAMES + 1, dtype=np.float32)
    trajectories = np.broadcast_to(frames[:, None, None], (1, NUM_FRAMES, 1, 8)).copy()
    window_starts = (5, 0, 3)
    baselines = evaluate_segment_baselines(
        trajectories,
        window_starts,
        context_length=CONTEXT_LENGTH,
        rollout_steps=ROLLOUT_STEPS,
    )
    expected = [1 / (start + CONTEXT_LENGTH + 1) for start in window_starts]
    assert np.allclose(baselines.identity.l2_onestep[0], expected, atol=ATOL)


def test_a_window_past_the_last_frame_is_rejected():
    last_start = NUM_FRAMES - CONTEXT_LENGTH - ROLLOUT_STEPS
    with pytest.raises(ValueError, match="window_starts"):
        evaluate_segment_baselines(
            _traveling_sines(),
            (0, last_start + 1),
            context_length=CONTEXT_LENGTH,
            rollout_steps=ROLLOUT_STEPS,
        )


def test_match_trajectories_finds_partners_in_a_shuffled_subset():
    # Four trajectories in two coefficient groups. The partner set holds three of
    # them in another order, with a perturbation below the tolerance.
    fields = np.asarray([[0.0, 1.0], [2.0, 3.0], [0.0, 1.0], [5.0, 6.0]])
    coefficients = np.asarray([[1.0], [1.0], [2.0], [2.0]])
    order = [2, 0, 3]
    partner_fields = fields[order] + 1e-9
    partners = match_trajectories(
        fields, coefficients, partner_fields, coefficients[order], atol=1e-6
    )
    # Trajectory 1 has no partner. Trajectories 0 and 2 have equal fields and are
    # told apart by their coefficients.
    assert partners.tolist() == [1, -1, 0, 2]


def test_match_trajectories_rejects_a_shared_partner():
    fields = np.zeros((2, 3))
    coefficients = np.ones((2, 1))
    with pytest.raises(ValueError, match="share a partner"):
        match_trajectories(fields, coefficients, fields[:1], coefficients[:1], atol=0.0)


def test_mean_by_bin():
    values = np.asarray([[1.0, 3.0, 10.0, 7.0, 100.0], [2.0, 4.0, 20.0, 8.0, 200.0]])
    # The last entry lies outside the bins, and bin 2 is empty.
    means, counts = mean_by_bin(values, np.asarray([0, 0, 1, 3, -1]), num_bins=4)
    assert counts.tolist() == [2, 1, 0, 1]
    assert np.allclose(means[:, [0, 1, 3]], [[2.0, 10.0, 7.0], [3.0, 20.0, 8.0]])
    assert np.isnan(means[:, 2]).all()
