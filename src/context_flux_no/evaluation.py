"""Evaluation of trained models on a set of test trajectories.

This module computes the rollout diagnostics of `context_flux_no.metrics` (D1 to D6 of
the ICLR 2027 rebuttal plan, action 2b). `evaluate_rollouts` and `evaluate_references`
run on the accelerator, and `summarize_rollouts` reduces their per-step arrays on the
host. `scripts/evaluation/rollout_diagnostics.py` applies them to the test sets of the
paper.

The module also computes the errors of every segment of a test set (actions 5a and
5b). `evaluate_segments` and `evaluate_segment_baselines` run on the accelerator, and
`match_trajectories` and `mean_by_bin` group their results on the host.
`scripts/evaluation/segment_errors.py` applies them to the shallow-water test sets.

`max_cubic_wave_speed` and `equal_count_bins` group the segments of a 1D cubic test set
by their CFL number (action 7c). `scripts/evaluation/context_length.py` applies them.

The last part changes what the encoder of an HFluxNO model reads (actions 3a and 3e).
`rollout_with_refresh` regenerates the flux networks every few steps, and
`evaluate_context_interventions` generates them from a shuffled, a static or a donor
context. `scripts/evaluation/flux_refresh.py` and
`scripts/evaluation/context_interventions.py` apply them.

`evaluate_effective_flux` evaluates the generated flux network on constant states and
compares it with the true flux (action 3b), and `evaluate_time_steps` advances one
step of another size than the training step (action 7c).
`scripts/evaluation/effective_flux.py` and `scripts/evaluation/time_step.py` apply
them.
"""

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int
from scipy.optimize import lsq_linear

from .metrics import (
    error_power_spectrum,
    first_true_step,
    ideal_gas_pressure,
    rollout_diagnostics,
    rollout_references,
    RolloutDiagnostics,
    RolloutReferences,
    segment_errors,
    SegmentErrors,
    spatial_mean_field,
    valid_horizon,
)
from .models.multiphysics.abstract import AbstractMultiphysicsOperator
from .models.multiphysics.hyperfluxfno import HyperFluxFNOLocal
from .nn.operators.fourier_utils import append_grid_channels


# A function that returns, for every step of a predicted rollout, the minimum of each
# quantity that must stay positive. Its second argument is one coefficient of the
# trajectory, which the Euler equations need and shallow water ignores.
MinimaFn = Callable[
    [Float[Array, " steps channels *spatial_dims"], Float[Array, ""]],
    Float[Array, " steps quantities"],
]


def shallow_water_minima(
    u_pred: Float[Array, " steps 2 x"], coefficient: Float[Array, ""]
) -> Float[Array, " steps 1"]:
    """Return the minimum water depth at every step of a predicted rollout.

    The channel order is `(h, h * u)`, as in
    `context_flux_no.simulations.pde.shallow_water`. `coefficient` is not used.
    """
    del coefficient
    return jnp.min(u_pred[:, 0], axis=-1)[:, None]


def euler_minima(
    u_pred: Float[Array, " steps channels *spatial_dims"], gamma: Float[Array, ""]
) -> Float[Array, " steps 2"]:
    """Return the minimum density and pressure at every step of a predicted rollout.

    The channel order is `(rho, E, m_1, ..., m_d)`, the order in which
    `context_flux_no.data.sources.WellDatasetSourceBase` packs the fields of the Euler
    datasets. `gamma` is the ratio of specific heats of the trajectory.
    """
    spatial_axes = tuple(range(1, u_pred.ndim - 1))
    density, energy, momentum = u_pred[:, 0], u_pred[:, 1], u_pred[:, 2:]
    pressure = jax.vmap(ideal_gas_pressure, in_axes=(0, 0, 0, None))(
        density, momentum, energy, gamma
    )
    return jnp.stack(
        (jnp.min(density, axis=spatial_axes), jnp.min(pressure, axis=spatial_axes)),
        axis=-1,
    )


@dataclasses.dataclass(frozen=True)
class RolloutEvaluation:
    """The result of rolling one model out over a set of trajectories.

    The result lives on the host and holds NumPy arrays, the fields of `diagnostics`
    included.

    **Attributes:**

    - `diagnostics`: The per-step diagnostics of every trajectory.
    - `minima`: The per-step minima of the quantities that must stay positive, or
        `None` when no `minima_fn` was given.
    - `error_spectrum`: The error spectrum (D6) at the requested steps, averaged over
        the trajectories whose prediction is finite at that step. `None` when no
        spectrum was requested.
    - `num_finite_spectra`: The number of trajectories in that average, per step.
    """

    diagnostics: RolloutDiagnostics
    minima: Float[np.ndarray, " trajectories steps quantities"] | None
    error_spectrum: Float[np.ndarray, " spectrum_steps wavenumbers"] | None
    num_finite_spectra: Float[np.ndarray, " spectrum_steps"] | None


def _rollout_one(
    model: AbstractMultiphysicsOperator,
    u: Float[Array, " frames channels *spatial_dims"],
    coefficient: Float[Array, ""],
    model_args: Any,
    context_length: int,
    area_thresholds: Float[Array, " thresholds"],
    spectrum_indices: tuple[int, ...],
    minima_fn: MinimaFn | None,
) -> tuple[RolloutDiagnostics, Array | None, Array | None]:
    """Roll `model` out from the first `context_length` frames of `u` and compare the
    prediction with the remaining frames."""
    context, u_data = u[:context_length], u[context_length:]
    u_pred = model.rollout(context, model_args, num_steps=len(u_data))[0]

    diagnostics = rollout_diagnostics(context[-1], u_pred, u_data, area_thresholds)
    minima = None if minima_fn is None else minima_fn(u_pred, coefficient)
    spectrum = None
    if spectrum_indices:
        indices = jnp.asarray(spectrum_indices)
        spectrum = jax.vmap(error_power_spectrum)(
            u_pred[indices, 0], u_data[indices, 0]
        )
    return diagnostics, minima, spectrum


def evaluate_rollouts(
    model: AbstractMultiphysicsOperator,
    trajectories: Float[np.ndarray, " trajectories frames channels *spatial_dims"],
    model_args: Any,
    *,
    context_length: int,
    area_thresholds: tuple[float, ...],
    spectrum_steps: tuple[int, ...] = (),
    minima_fn: MinimaFn | None = None,
    coefficients: Float[np.ndarray, " trajectories"] | None = None,
    chunk_size: int = 1000,
    batch_size: int = 1000,
) -> RolloutEvaluation:
    """Roll a model out over every trajectory and compute the rollout diagnostics.

    Each trajectory gives one rollout. Its first `context_length` frames are the
    context, and the model predicts all remaining frames autoregressively.

    **Arguments:**

    - `model`: The model, already in inference mode.
    - `trajectories`: The test trajectories. The array stays on the host.
    - `model_args`: The `args` of `model.rollout`, such as `(dt, dx)`.
    - `context_length`: The number of context frames.
    - `area_thresholds`: The thresholds of the error area fraction.
    - `spectrum_steps`: The rollout steps, counted from 1, at which the error spectrum
        of the first channel is computed. Empty for no spectrum.
    - `minima_fn`: The function that returns the minima of the quantities that must
        stay positive, such as `euler_minima`. `None` for none.
    - `coefficients`: One coefficient per trajectory, passed to `minima_fn`. `None` is
        allowed only when `minima_fn` ignores it.
    - `chunk_size`: The number of trajectories that are moved to the device at once.
    - `batch_size`: The number of rollouts that run in parallel on the device.

    **Returns:**

    A `RolloutEvaluation` whose arrays are NumPy arrays on the host.
    """
    num_trajectories, num_frames = trajectories.shape[:2]
    num_steps = num_frames - context_length
    if num_steps < 1:
        raise ValueError(
            f"Trajectories of {num_frames} frames leave no step after a context of "
            f"{context_length} frames."
        )
    if any(step < 1 or step > num_steps for step in spectrum_steps):
        raise ValueError(
            f"spectrum_steps={spectrum_steps} must lie within 1 and {num_steps}."
        )
    if coefficients is None:
        coefficients = np.zeros(num_trajectories, dtype=trajectories.dtype)
    if len(coefficients) != num_trajectories:
        raise ValueError(
            f"Got {len(coefficients)} coefficients for {num_trajectories} trajectories."
        )
    spectrum_indices = tuple(step - 1 for step in spectrum_steps)
    thresholds = jnp.asarray(area_thresholds)

    @eqx.filter_jit
    def evaluate_chunk(model, u_chunk, coefficient_chunk):
        return jax.lax.map(
            lambda inputs: _rollout_one(
                model,
                inputs[0],
                inputs[1],
                model_args,
                context_length,
                thresholds,
                spectrum_indices,
                minima_fn,
            ),
            (u_chunk, coefficient_chunk),
            batch_size=batch_size,
        )

    diagnostics_chunks, minima_chunks = [], []
    spectrum_sum, spectrum_count = 0.0, 0.0
    for start in range(0, num_trajectories, chunk_size):
        stop = start + chunk_size
        diagnostics, minima, spectrum = evaluate_chunk(
            model,
            jnp.asarray(trajectories[start:stop]),
            jnp.asarray(coefficients[start:stop]),
        )
        diagnostics_chunks.append(jax.tree.map(np.asarray, diagnostics))
        if minima is not None:
            minima_chunks.append(np.asarray(minima))
        if spectrum is not None:
            spectrum = np.asarray(spectrum)
            is_finite = np.all(np.isfinite(spectrum), axis=-1, keepdims=True)
            spectrum_sum = spectrum_sum + np.where(is_finite, spectrum, 0.0).sum(axis=0)
            spectrum_count = spectrum_count + is_finite[..., 0].sum(axis=0)

    has_spectrum = bool(spectrum_indices)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_spectrum = spectrum_sum / spectrum_count[:, None] if has_spectrum else None
    return RolloutEvaluation(
        diagnostics=jax.tree.map(lambda *xs: np.concatenate(xs), *diagnostics_chunks),
        minima=np.concatenate(minima_chunks) if minima_chunks else None,
        error_spectrum=mean_spectrum,
        num_finite_spectra=spectrum_count if has_spectrum else None,
    )


def evaluate_references(
    trajectories: Float[np.ndarray, " trajectories frames channels *spatial_dims"],
    *,
    context_length: int,
    chunk_size: int = 1000,
) -> RolloutReferences:
    """Compute the model-independent reference quantities of every rollout.

    The rollouts are those of `evaluate_rollouts`: the context is the first
    `context_length` frames, and the remaining frames are the reference.
    """

    @jax.jit
    def evaluate_chunk(u_chunk):
        return jax.vmap(rollout_references)(
            u_chunk[:, context_length - 1], u_chunk[:, context_length:]
        )

    chunks = [
        jax.tree.map(
            np.asarray,
            evaluate_chunk(jnp.asarray(trajectories[start : start + chunk_size])),
        )
        for start in range(0, trajectories.shape[0], chunk_size)
    ]
    return jax.tree.map(lambda *xs: np.concatenate(xs), *chunks)


def _fraction_by_step(
    flags: Bool[np.ndarray, " *batch steps"], steps: tuple[int, ...]
) -> dict[int, float]:
    """Return the fraction of rows that are true at any step up to each of `steps`."""
    first = first_true_step(flags)
    return {step: float(np.mean(first <= step)) for step in steps}


def _median_over_finite(
    values: Float[np.ndarray, " *batch steps"],
    nonfinite: Bool[np.ndarray, " *batch steps"],
    steps: tuple[int, ...],
) -> dict[int, float]:
    """Return the median of `values` at each of `steps` over the finite rollouts."""
    medians = {}
    for step in steps:
        column = values[..., step - 1][~nonfinite[..., step - 1]]
        column = column[np.isfinite(column)]
        medians[step] = float(np.median(column)) if column.size else float("nan")
    return medians


def summarize_rollouts(
    l2: Float[np.ndarray, " *batch trajectories steps"],
    nonfinite: Bool[np.ndarray, " *batch trajectories steps"],
    area_fraction: Float[np.ndarray, " *batch trajectories steps thresholds"],
    total_variation: Float[np.ndarray, " *batch trajectories steps"],
    references: RolloutReferences,
    *,
    exclude_constant_references: bool,
    minima: Float[np.ndarray, " *batch trajectories steps quantities"] | None = None,
    horizon: int = 80,
    report_steps: tuple[int, ...] = (20, 40, 80),
) -> dict[str, Any]:
    """Reduce the per-step arrays of one model to the numbers of D2 to D5.

    The leading `*batch` axes, such as the seeds of one model, are pooled with the
    trajectories. Only the first `horizon` steps are used.

    A trajectory has a constant reference when the total variation of its reference is
    zero at every step of the horizon. This happens for a scalar law whose initial
    condition is a constant state, such as a periodic step function with one jump.
    Both baselines then have zero error and the reference has zero range, so every
    model error counts as a crossing of the identity baseline at step 1 and as
    diverged (D2 and D3), the area fraction is 1 wherever the prediction differs (D4),
    and the total-variation ratio is infinite (D5).

    **Arguments:**

    - `l2`, `nonfinite`, `area_fraction`, `total_variation`: The fields of
        `RolloutDiagnostics`.
    - `references`: The reference quantities of the same trajectories.
    - `exclude_constant_references`: Whether the trajectories with a constant
        reference are left out of every number of the summary.
    - `minima`: The minima of the quantities that must stay positive, or `None`.
    - `horizon`: The number of rollout steps that enter the summary.
    - `report_steps`: The steps at which the failure fractions, the area fraction and
        the total-variation ratio are reported.

    **Returns:**

    A dictionary with the following entries.

    - `num_rollouts`: The number of rollouts that enter the summary.
    - `num_constant_references`: The number of trajectories with a constant reference,
        whether or not they are left out.
    - `valid_horizon_quartiles`: The 25th, 50th and 75th percentile of the valid
        horizon (D2), where a rollout that stays below the identity baseline counts as
        `horizon`.
    - `fraction_never_crossed`: The fraction of rollouts that stay below the identity
        baseline.
    - `fraction_crossed_at_step_1`: The fraction that is no better than the identity
        baseline at the first step.
    - `fraction_nonfinite`, `fraction_diverged`, `fraction_nonpositive`: The fraction
        of rollouts affected at any step up to each of `report_steps` (D3). A
        non-finite step counts as diverged. `fraction_nonpositive` is `None` without
        `minima`.
    - `median_area_fraction`: The median error area fraction per threshold (D4), over
        the rollouts that are finite at the step.
    - `median_total_variation_ratio`: The median of the total variation of the
        prediction over that of the reference (D5), over the same rollouts.
    """
    num_steps = l2.shape[-1]
    if num_steps < horizon:
        raise ValueError(f"The rollouts have {num_steps} steps, fewer than {horizon}.")
    if any(step > horizon for step in report_steps):
        raise ValueError(f"report_steps={report_steps} exceed the horizon {horizon}.")

    is_constant = np.all(references.total_variation[..., :horizon] == 0.0, axis=-1)
    keep = ~is_constant if exclude_constant_references else np.ones_like(is_constant)

    l2 = l2[..., keep, :horizon]
    nonfinite = nonfinite[..., keep, :horizon]
    area_fraction = area_fraction[..., keep, :, :]
    l2_identity = references.l2_identity[keep, :horizon]
    l2_spatial_mean = references.l2_spatial_mean[keep, :horizon]

    horizons = valid_horizon(l2, l2_identity, nonfinite)
    with np.errstate(invalid="ignore", divide="ignore"):
        diverged = (l2 > l2_spatial_mean) | nonfinite | ~np.isfinite(l2)
        ratio = (
            total_variation[..., keep, :horizon]
            / references.total_variation[keep, :horizon]
        )
    if minima is not None:
        minima = minima[..., keep, :, :]

    summary = {
        "num_rollouts": int(horizons.size),
        "num_constant_references": int(is_constant.sum()),
        "valid_horizon_quartiles": [
            float(q) for q in np.percentile(np.minimum(horizons, horizon), [25, 50, 75])
        ],
        "fraction_never_crossed": float(np.mean(horizons > horizon)),
        "fraction_crossed_at_step_1": float(np.mean(horizons == 1)),
        "fraction_nonfinite": _fraction_by_step(nonfinite, report_steps),
        "fraction_diverged": _fraction_by_step(diverged, report_steps),
        "fraction_nonpositive": None,
        "median_area_fraction": [
            _median_over_finite(
                area_fraction[..., :horizon, i], nonfinite, report_steps
            )
            for i in range(area_fraction.shape[-1])
        ],
        "median_total_variation_ratio": _median_over_finite(
            ratio, nonfinite, report_steps
        ),
    }
    if minima is not None:
        with np.errstate(invalid="ignore"):
            nonpositive = np.any(minima[..., :horizon, :] <= 0.0, axis=-1)
        summary["fraction_nonpositive"] = _fraction_by_step(nonpositive, report_steps)
    return summary


def summarize_conservation_drift(
    conservation_drift: Float[np.ndarray, " trajectories steps channels"],
    *,
    horizon: int = 80,
) -> dict[str, Any]:
    """Reduce the conservation drift of one checkpoint to three numbers per channel.

    The three numbers are those of the conservation result of the submission. Each is
    a mean over the trajectories whose drift is finite at every step of the horizon.
    The caller applies the function to every seed of a model and reports the mean and
    the standard deviation over the seeds, as for the other tables of the paper.

    **Arguments:**

    - `conservation_drift`: The field of `RolloutDiagnostics` or `RolloutReferences`
        of that name.
    - `horizon`: The number of rollout steps that enter the summary.

    **Returns:**

    A dictionary with the following entries.

    - `mean`: The mean of the drift over the steps and the trajectories.
    - `final`: The mean over the trajectories of the drift at the last step.
    - `worst`: The mean over the trajectories of the largest drift of a trajectory.
    - `fraction_finite`: The fraction of trajectories that enter these means, per
        channel.
    """
    num_steps = conservation_drift.shape[1]
    if num_steps < horizon:
        raise ValueError(f"The rollouts have {num_steps} steps, fewer than {horizon}.")
    drift = conservation_drift[:, :horizon]
    is_finite = np.all(np.isfinite(drift), axis=1)
    count = is_finite.sum(axis=0)

    def mean_over_finite(
        per_trajectory: Float[np.ndarray, " trajectories channels"],
    ) -> Float[np.ndarray, " channels"]:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(is_finite, per_trajectory, 0.0).sum(axis=0) / count

    with np.errstate(invalid="ignore"):
        return {
            "mean": mean_over_finite(drift.mean(axis=1)),
            "final": mean_over_finite(drift[:, -1]),
            "worst": mean_over_finite(drift.max(axis=1)),
            "fraction_finite": count / drift.shape[0],
        }


# --- Segment errors (ICLR 2027 rebuttal, actions 5a and 5b) -------------------------


class SegmentBaselines(eqx.Module):
    """Errors of the two baselines on the segments of a test set.

    **Attributes:**

    - `identity`: The errors of the identity baseline, which holds the last context
        frame over the whole segment.
    - `spatial_mean`: The errors of the spatial-mean baseline, which holds the cell
        mean of each channel of the last context frame.
    """

    identity: SegmentErrors
    spatial_mean: SegmentErrors


def _check_window_starts(
    window_starts: tuple[int, ...], num_frames: int, segment_length: int
) -> None:
    if not window_starts:
        raise ValueError("window_starts is empty.")
    last_start = num_frames - segment_length
    if min(window_starts) < 0 or max(window_starts) > last_start:
        raise ValueError(
            f"window_starts must lie within 0 and {last_start} for {num_frames} "
            f"frames and segments of {segment_length} frames, got {window_starts}."
        )


def _segments_of(
    u: Float[Array, " frames channels *spatial_dims"],
    window_starts: Int[Array, " windows"],
    segment_length: int,
) -> Float[Array, " windows {segment_length} channels *spatial_dims"]:
    """Return the segments of one trajectory that begin at `window_starts`."""
    return u[window_starts[:, None] + jnp.arange(segment_length)]


def _map_over_chunks(evaluate_chunk, trajectories: np.ndarray, chunk_size: int):
    """Apply a jitted function to chunks of trajectories and join the results on the
    host."""
    chunks = [
        jax.tree.map(
            np.asarray,
            evaluate_chunk(jnp.asarray(trajectories[start : start + chunk_size])),
        )
        for start in range(0, trajectories.shape[0], chunk_size)
    ]
    return jax.tree.map(lambda *xs: np.concatenate(xs), *chunks)


def evaluate_segments(
    model: AbstractMultiphysicsOperator,
    trajectories: Float[np.ndarray, " trajectories frames channels *spatial_dims"],
    window_starts: tuple[int, ...],
    model_args: Any,
    *,
    context_length: int,
    rollout_steps: int,
    chunk_size: int = 1000,
    batch_size: int = 16,
) -> SegmentErrors:
    """Compute the errors of a model on every segment of every trajectory.

    A segment begins at one of `window_starts`. Its first `context_length` frames are
    the context, and the model predicts the next `rollout_steps` frames
    autoregressively. This is the evaluation of the submission, which
    `notebooks/evaluation/evaluate_shallow.ipynb` runs with every window start, except
    that the errors of the segments are returned and not averaged.

    **Arguments:**

    - `model`: The model, already in inference mode.
    - `trajectories`: The test trajectories. The array stays on the host.
    - `window_starts`: The frame indices at which segments begin.
    - `model_args`: The `args` of `model.rollout`, such as `(dt, dx)`.
    - `context_length`: The number of context frames.
    - `rollout_steps`: The number of predicted frames.
    - `chunk_size`: The number of trajectories that are moved to the device at once.
    - `batch_size`: The number of trajectories whose segments run in parallel on the
        device. The number of parallel rollouts is `batch_size * len(window_starts)`.

    **Returns:**

    A `SegmentErrors` whose fields are NumPy arrays of shape `(trajectories, windows)`.
    """
    segment_length = context_length + rollout_steps
    _check_window_starts(window_starts, trajectories.shape[1], segment_length)
    starts = jnp.asarray(window_starts)

    def one_segment(model, segment):
        context, u_data = segment[:context_length], segment[context_length:]
        u_pred = model.rollout(context, model_args, num_steps=rollout_steps)[0]
        return segment_errors(u_pred, u_data)

    @eqx.filter_jit
    def evaluate_chunk(model, u_chunk):
        return jax.lax.map(
            lambda u: jax.vmap(lambda segment: one_segment(model, segment))(
                _segments_of(u, starts, segment_length)
            ),
            u_chunk,
            batch_size=batch_size,
        )

    return _map_over_chunks(
        lambda u_chunk: evaluate_chunk(model, u_chunk), trajectories, chunk_size
    )


def evaluate_segment_baselines(
    trajectories: Float[np.ndarray, " trajectories frames channels *spatial_dims"],
    window_starts: tuple[int, ...],
    *,
    context_length: int,
    rollout_steps: int,
    chunk_size: int = 1000,
) -> SegmentBaselines:
    """Compute the errors of the two baselines on the segments of `evaluate_segments`.

    **Returns:**

    A `SegmentBaselines` whose arrays have the shape `(trajectories, windows)`.
    """
    segment_length = context_length + rollout_steps
    _check_window_starts(window_starts, trajectories.shape[1], segment_length)
    starts = jnp.asarray(window_starts)

    def one_segment(segment):
        last, u_data = segment[context_length - 1], segment[context_length:]
        return SegmentBaselines(
            identity=segment_errors(jnp.broadcast_to(last, u_data.shape), u_data),
            spatial_mean=segment_errors(
                jnp.broadcast_to(spatial_mean_field(last), u_data.shape), u_data
            ),
        )

    @jax.jit
    def evaluate_chunk(u_chunk):
        return jax.vmap(
            lambda u: jax.vmap(one_segment)(_segments_of(u, starts, segment_length))
        )(u_chunk)

    return _map_over_chunks(evaluate_chunk, trajectories, chunk_size)


def match_trajectories(
    fields: Float[np.ndarray, " n *shape"],
    coefficients: Float[np.ndarray, " n params"],
    partner_fields: Float[np.ndarray, " m *shape"],
    partner_coefficients: Float[np.ndarray, " m params"],
    *,
    atol: float,
) -> Int[np.ndarray, " n"]:
    """Find, for each trajectory of one test set, its partner in a paired test set.

    Two trajectories are partners when their coefficients are equal and their `fields`
    agree within `atol` in every entry. `fields` is a quantity that pairing leaves
    unchanged, such as the initial momentum of two shallow-water sets that differ in
    the initial depth only.

    **Returns:**

    The index of the partner of each trajectory, or -1 for a trajectory without one.

    **Raises:**

    `ValueError` if two trajectories share a partner, because the sets are then not
    paired one to one.
    """
    by_coefficients: dict[bytes, list[int]] = {}
    for index, row in enumerate(np.ascontiguousarray(partner_coefficients)):
        by_coefficients.setdefault(row.tobytes(), []).append(index)

    partners = np.full(len(fields), -1, dtype=np.int64)
    for index, row in enumerate(np.ascontiguousarray(coefficients)):
        candidates = np.asarray(by_coefficients.get(row.tobytes(), []), dtype=np.int64)
        if candidates.size == 0:
            continue
        distance = np.abs(partner_fields[candidates] - fields[index]).reshape(
            len(candidates), -1
        )
        distance = distance.max(axis=1)
        if distance.min() <= atol:
            partners[index] = candidates[distance.argmin()]

    matched = partners[partners >= 0]
    if len(np.unique(matched)) != len(matched):
        raise ValueError("Two trajectories share a partner; the sets are not paired.")
    return partners


def mean_by_bin(
    values: Float[np.ndarray, " *batch n"],
    bin_indices: Int[np.ndarray, " n"],
    num_bins: int,
) -> tuple[Float[np.ndarray, " *batch {num_bins}"], Int[np.ndarray, " {num_bins}"]]:
    """Average `values` over the entries of each bin.

    **Arguments:**

    - `values`: The values, with the binned axis last.
    - `bin_indices`: The bin of each entry of that axis. An entry with an index outside
        0 to `num_bins - 1` belongs to no bin.
    - `num_bins`: The number of bins.

    **Returns:**

    The mean per bin, which is NaN for an empty bin, and the number of entries per bin.
    A non-finite value makes the mean of its bin non-finite.
    """
    in_range = (bin_indices >= 0) & (bin_indices < num_bins)
    counts = np.bincount(bin_indices[in_range], minlength=num_bins)
    one_hot = bin_indices[:, None] == np.arange(num_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = (
            np.stack(
                [values[..., one_hot[:, b]].sum(axis=-1) for b in range(num_bins)],
                axis=-1,
            )
            / counts
        )
    return means, counts


# --- CFL number of a segment (ICLR 2027 rebuttal, action 7c) -------------------------


def max_cubic_wave_speed(
    coefficients: Float[np.ndarray, " *#batch 3"],
    lower: Float[np.ndarray, " *#batch"],
    upper: Float[np.ndarray, " *#batch"],
) -> Float[np.ndarray, " *batch"]:
    """Return the largest wave speed of a cubic flux over an interval of states.

    The flux is f(u) = a u^3 + b u^2 + c u, as in
    `context_flux_no.simulations.pde.cubic`, and the wave speed is |f'(u)| with
    f'(u) = 3 a u^2 + 2 b u + c. The maximum of |f'| over `[lower, upper]` is at an end
    of the interval or at the vertex u = -b / (3 a) of the parabola, when the vertex
    lies inside. Multiplied by dt / dx, the result is the CFL number of a segment whose
    last context frame takes values between `lower` and `upper`.

    **Arguments:**

    - `coefficients`: The coefficients `(a, b, c)` along the last axis.
    - `lower`: The lower end of the interval. It broadcasts against the leading axes
        of `coefficients`.
    - `upper`: The upper end of the interval, not below `lower`.
    """
    a, b, c = np.moveaxis(np.asarray(coefficients), -1, 0)
    lower, upper = np.asarray(lower), np.asarray(upper)

    def speed(u):
        return np.abs((3 * a * u + 2 * b) * u + c)

    with np.errstate(divide="ignore", invalid="ignore"):
        vertex = -b / (3 * a)
    is_inside = np.isfinite(vertex) & (vertex > lower) & (vertex < upper)
    # Outside the interval the vertex is replaced by an end, which changes nothing.
    at_vertex = speed(np.where(is_inside, vertex, lower))
    return np.maximum(np.maximum(speed(lower), speed(upper)), at_vertex)


def equal_count_bins(
    values: Float[np.ndarray, " n"], num_bins: int
) -> tuple[Int[np.ndarray, " n"], Float[np.ndarray, " {num_bins}+1"]]:
    """Assign each value to one of `num_bins` bins that hold equally many values.

    The bin edges are the quantiles of the finite values at 0, 1 / num_bins, ..., 1.
    A bin contains its lower edge, and the last bin also contains its upper edge.
    Values that are tied at an edge all fall into one bin, so the counts can differ.

    **Returns:**

    The bin index of each value, in the form that `mean_by_bin` takes, and the bin
    edges. A non-finite value has the index -1.
    """
    values = np.asarray(values)
    is_finite = np.isfinite(values)
    if not is_finite.any():
        raise ValueError("No value is finite, so the bins have no edges.")
    edges = np.quantile(values[is_finite], np.linspace(0.0, 1.0, num_bins + 1))
    bin_indices = np.searchsorted(edges, values, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    return np.where(is_finite, bin_indices, -1), edges


def cubic_wave_speed_distance(
    coefficients: Float[np.ndarray, " *#batch 3"],
    other_coefficients: Float[np.ndarray, " *#batch 3"],
    lower: Float[np.ndarray, " *#batch"],
    upper: Float[np.ndarray, " *#batch"],
    num_points: int = 129,
) -> Float[np.ndarray, " *batch"]:
    """Return the relative L2 distance between the derivatives of two cubic fluxes
    over an interval of states.

    With f the flux of `coefficients` and g the flux of `other_coefficients`, the
    result is ||g' - f'|| / ||f'||, where both norms are L2 norms over
    `[lower, upper]`. The integrals use the trapezoidal rule on `num_points` states.
    An interval of zero length gives |g'(u) - f'(u)| / |f'(u)| at its one state. The
    result is not finite where f' vanishes on the whole interval.

    **Arguments:**

    - `coefficients`: The coefficients `(a, b, c)` of the reference flux
        f(u) = a u^3 + b u^2 + c u, along the last axis.
    - `other_coefficients`: The coefficients of the flux that is compared with it.
    - `lower`: The lower end of the interval.
    - `upper`: The upper end of the interval.
    - `num_points`: The number of states of the trapezoidal rule.
    """
    lower = np.asarray(lower)[..., None]
    upper = np.asarray(upper)[..., None]
    states = lower + (upper - lower) * np.linspace(0.0, 1.0, num_points)

    def derivative(coefficients_):
        a, b, c = np.moveaxis(np.asarray(coefficients_)[..., None, :], -1, 0)
        return (3 * a * states + 2 * b) * states + c

    reference = derivative(coefficients)
    difference = derivative(other_coefficients) - reference
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(
            np.trapezoid(difference**2, axis=-1) / np.trapezoid(reference**2, axis=-1)
        )


# --- Flux networks held fixed or generated from another context (actions 3a, 3e) -----

# The contexts that `evaluate_context_interventions` feeds to the encoder before the
# donor contexts, in the order of the last axis of its result.
INTERVENTIONS = ("true", "shuffled", "static")


class RefreshErrors(eqx.Module):
    """Errors of one rollout whose flux networks are regenerated every few steps.

    **Attributes:**

    - `l2_rollout`: The relative l2 error of all predicted frames, with one norm over
        time, channels and space, as in `SegmentErrors`.
    - `l_inf_rollout`: The relative l-infinity error of all predicted frames.
    - `l2_final`: The relative l2 error of the last predicted frame.
    - `l_inf_final`: The relative l-infinity error of the last predicted frame.
    """

    l2_rollout: Float[Array, ""]
    l_inf_rollout: Float[Array, ""]
    l2_final: Float[Array, ""]
    l_inf_final: Float[Array, ""]


def rollout_with_flux_models(
    model: HyperFluxFNOLocal,
    flux_models: tuple[eqx.Module, ...],
    u: Float[Array, " channels *spatial_dims"],
    args: tuple[float, ...],
    num_steps: int,
) -> Float[Array, " {num_steps} channels *spatial_dims"]:
    """Advance a state by `num_steps` steps with one set of flux networks.

    **Arguments:**

    - `model`: The model, already in inference mode.
    - `flux_models`: The flux networks, as `model.generate_flux_models` returns them.
    - `u`: The state before the first step.
    - `args`: The `args` of `model.step`, such as `(dt, dx)`.
    - `num_steps`: The number of steps.

    **Returns:**

    The states after each step.
    """

    def scan_fn(u_, _):
        u_next = model.step(flux_models, u_, args)
        return u_next, u_next

    return jax.lax.scan(scan_fn, u, None, length=num_steps)[1]


def rollout_with_refresh(
    model: HyperFluxFNOLocal,
    context: Float[Array, " time channels *spatial_dims"],
    args: tuple[float, ...],
    num_steps: int,
    refresh_every: int,
) -> Float[Array, " {num_steps} channels *spatial_dims"]:
    """Roll a model out and regenerate its flux networks every `refresh_every` steps.

    The flux networks are generated from the latest `len(context)` frames, of which
    the newest are predictions, and are held for `refresh_every` steps. With
    `refresh_every = 1` this is `model.rollout`. With `refresh_every >= num_steps` the
    flux networks are generated once from `context`.

    **Arguments:**

    - `model`: The model, already in inference mode.
    - `context`: The context frames.
    - `args`: The `args` of `model.step`, such as `(dt, dx)`.
    - `num_steps`: The number of predicted frames.
    - `refresh_every`: The number of steps for which one set of flux networks is used.
    """
    if refresh_every < 1:
        raise ValueError(f"refresh_every must be at least 1, got {refresh_every}.")
    context_length = context.shape[0]

    def block_fn(window, _):
        flux_models = model.generate_flux_models(window)
        u_block = rollout_with_flux_models(
            model, flux_models, window[-1], args, refresh_every
        )
        window = jnp.concatenate((window, u_block))[-context_length:]
        return window, u_block

    # The last block may run past `num_steps`. Its extra steps are computed and
    # dropped, which keeps every block the same length for `lax.scan`.
    num_blocks = math.ceil(num_steps / refresh_every)
    blocks = jax.lax.scan(block_fn, context, None, length=num_blocks)[1]
    return blocks.reshape(-1, *blocks.shape[2:])[:num_steps]


def evaluate_refresh_rollouts(
    model: HyperFluxFNOLocal,
    trajectories: Float[np.ndarray, " trajectories frames channels *spatial_dims"],
    window_starts: tuple[int, ...],
    model_args: tuple[float, ...],
    *,
    context_length: int,
    rollout_steps: int,
    refresh_every: tuple[int, ...],
    chunk_size: int = 1000,
    batch_size: int = 16,
) -> RefreshErrors:
    """Compute the errors of `rollout_with_refresh` on every segment of every
    trajectory, for several refresh intervals.

    The segments are those of `evaluate_segments`.

    **Arguments:**

    - `model`: The model, already in inference mode.
    - `trajectories`: The test trajectories. The array stays on the host.
    - `window_starts`: The frame indices at which segments begin.
    - `model_args`: The `args` of `model.step`, such as `(dt, dx)`.
    - `context_length`: The number of context frames.
    - `rollout_steps`: The number of predicted frames.
    - `refresh_every`: The refresh intervals.
    - `chunk_size`: The number of trajectories that are moved to the device at once.
    - `batch_size`: The number of trajectories whose segments run in parallel on the
        device.

    **Returns:**

    A `RefreshErrors` whose fields are NumPy arrays of shape
    `(trajectories, windows, len(refresh_every))`.
    """
    segment_length = context_length + rollout_steps
    _check_window_starts(window_starts, trajectories.shape[1], segment_length)
    starts = jnp.asarray(window_starts)

    def one_rollout(model, segment, interval):
        context, u_data = segment[:context_length], segment[context_length:]
        u_pred = rollout_with_refresh(
            model, context, model_args, rollout_steps, interval
        )
        errors = segment_errors(u_pred, u_data)
        final = segment_errors(u_pred[-1:], u_data[-1:])
        return RefreshErrors(
            l2_rollout=errors.l2_rollout,
            l_inf_rollout=errors.l_inf_rollout,
            l2_final=final.l2_onestep,
            l_inf_final=final.l_inf_onestep,
        )

    def one_segment(model, segment):
        per_interval = [
            one_rollout(model, segment, interval) for interval in refresh_every
        ]
        return jax.tree.map(lambda *xs: jnp.stack(xs), *per_interval)

    @eqx.filter_jit
    def evaluate_chunk(model, u_chunk):
        return jax.lax.map(
            lambda u: jax.vmap(lambda segment: one_segment(model, segment))(
                _segments_of(u, starts, segment_length)
            ),
            u_chunk,
            batch_size=batch_size,
        )

    return _map_over_chunks(
        lambda u_chunk: evaluate_chunk(model, u_chunk), trajectories, chunk_size
    )


def evaluate_context_interventions(
    model: HyperFluxFNOLocal,
    trajectories: Float[np.ndarray, " trajectories frames channels *spatial_dims"],
    donors: Int[np.ndarray, " trajectories num_donors"],
    permutations: Int[np.ndarray, " trajectories windows context_length"],
    window_starts: tuple[int, ...],
    model_args: tuple[float, ...],
    *,
    context_length: int,
    rollout_steps: int,
    chunk_size: int = 1000,
    batch_size: int = 16,
) -> SegmentErrors:
    """Compute the errors of a model whose flux networks are generated from a context
    other than the true one.

    On every segment the flux networks are generated once and held for the whole
    rollout, and the rollout starts from the true last context frame. Only the frames
    that the encoder reads change:

    - `true`: the context of the segment.
    - `shuffled`: the context of the segment in the order of `permutations`.
    - `static`: copies of the last context frame.
    - one entry per donor: the context of another trajectory at the same frames.

    **Arguments:**

    - `model`: The model, already in inference mode.
    - `trajectories`: The test trajectories. The array stays on the host.
    - `donors`: For every trajectory, the indices of its donor trajectories. An index
        of -1 marks a missing donor, whose errors are NaN.
    - `permutations`: For every segment, the order in which the encoder reads the
        context frames.
    - `window_starts`: The frame indices at which segments begin.
    - `model_args`: The `args` of `model.step`, such as `(dt, dx)`.
    - `context_length`: The number of context frames.
    - `rollout_steps`: The number of predicted frames.
    - `chunk_size`: The number of trajectories that are moved to the device at once.
    - `batch_size`: The number of trajectories whose segments run in parallel on the
        device.

    **Returns:**

    A `SegmentErrors` whose fields are NumPy arrays of shape
    `(trajectories, windows, len(INTERVENTIONS) + num_donors)`, with the entries of
    `INTERVENTIONS` first. `l2_rollout` and `l_inf_rollout` are the errors of the
    rollout with held flux networks.
    """
    segment_length = context_length + rollout_steps
    _check_window_starts(window_starts, trajectories.shape[1], segment_length)
    starts = jnp.asarray(window_starts)
    if permutations.shape != (
        trajectories.shape[0],
        len(window_starts),
        context_length,
    ):
        raise ValueError(
            f"permutations has the shape {permutations.shape}, expected "
            f"{(trajectories.shape[0], len(window_starts), context_length)}."
        )

    def one_segment(model, segment, donor_segments, permutation):
        context, u_data = segment[:context_length], segment[context_length:]
        encoder_inputs = jnp.concatenate(
            (
                jnp.stack(
                    (
                        context,
                        context[permutation],
                        jnp.broadcast_to(context[-1], context.shape),
                    )
                ),
                donor_segments[:, :context_length],
            )
        )

        def one_intervention(encoder_input):
            flux_models = model.generate_flux_models(encoder_input)
            u_pred = rollout_with_flux_models(
                model, flux_models, context[-1], model_args, rollout_steps
            )
            return segment_errors(u_pred, u_data)

        return jax.vmap(one_intervention)(encoder_inputs)

    def one_trajectory(model, u, u_donors, permutations_):
        donor_segments = jax.vmap(
            lambda u_donor: _segments_of(u_donor, starts, segment_length),
            out_axes=1,
        )(u_donors)
        return jax.vmap(lambda *xs: one_segment(model, *xs))(
            _segments_of(u, starts, segment_length), donor_segments, permutations_
        )

    @eqx.filter_jit
    def evaluate_chunk(model, u_chunk, donor_chunk, permutation_chunk):
        return jax.lax.map(
            lambda xs: one_trajectory(model, *xs),
            (u_chunk, donor_chunk, permutation_chunk),
            batch_size=batch_size,
        )

    chunks = []
    for start in range(0, trajectories.shape[0], chunk_size):
        chunk = slice(start, start + chunk_size)
        errors = evaluate_chunk(
            model,
            jnp.asarray(trajectories[chunk]),
            jnp.asarray(trajectories[np.clip(donors[chunk], 0, None)]),
            jnp.asarray(permutations[chunk]),
        )
        chunks.append(jax.tree.map(np.asarray, errors))
    errors = jax.tree.map(lambda *xs: np.concatenate(xs), *chunks)

    is_missing = np.concatenate(
        (np.zeros((len(donors), len(INTERVENTIONS)), dtype=bool), donors < 0), axis=1
    )[:, None, :]
    return jax.tree.map(lambda x: np.where(is_missing, np.nan, x), errors)


# --- Effective flux of the generated flux network (action 3b) ------------------------

# A function that evaluates a flux or its derivative for one set of coefficients.
FluxFn = Callable[
    [Float[Array, " params"], Float[Array, " *shape"]], Float[Array, " *shape"]
]


def cubic_flux(
    coefficients: Float[Array, " 3"], u: Float[Array, " *shape"]
) -> Float[Array, " *shape"]:
    """The flux f(u) = a u^3 + b u^2 + c u of
    `context_flux_no.simulations.pde.cubic`."""
    a, b, c = coefficients
    return ((a * u + b) * u + c) * u


def cubic_flux_derivative(
    coefficients: Float[Array, " 3"], u: Float[Array, " *shape"]
) -> Float[Array, " *shape"]:
    """The derivative f'(u) = 3 a u^2 + 2 b u + c of `cubic_flux`."""
    a, b, c = coefficients
    return (3 * a * u + 2 * b) * u + c


def sine_flux(
    coefficients: Float[Array, " 2"], u: Float[Array, " *shape"]
) -> Float[Array, " *shape"]:
    """The flux f(u) = a sin(b u) of `context_flux_no.simulations.pde.sine_1d`."""
    a, b = coefficients
    return a * jnp.sin(b * u)


def sine_flux_derivative(
    coefficients: Float[Array, " 2"], u: Float[Array, " *shape"]
) -> Float[Array, " *shape"]:
    """The derivative f'(u) = a b cos(b u) of `sine_flux`."""
    a, b = coefficients
    return a * b * jnp.cos(b * u)


# The flux and its derivative for the coefficient names that a test file stores.
FLUX_FAMILIES: dict[tuple[str, ...], tuple[FluxFn, FluxFn]] = {
    ("a", "b", "c"): (cubic_flux, cubic_flux_derivative),
    ("a", "b"): (sine_flux, sine_flux_derivative),
}


class EffectiveFlux(eqx.Module):
    """The generated flux network of one context, evaluated on constant states.

    A consistent numerical flux G satisfies G(u, ..., u) = f(u) + c for every constant
    state u, so its value on a constant field is the flux that it stands for, up to a
    constant. The flux network of the 1D model also reads the grid coordinate, so the
    value depends on the interface. The fields hold the mean over the interfaces.

    **Attributes:**

    - `states`: The constant states, equally spaced between the smallest and the
        largest value of the context.
    - `flux`: The mean over the interfaces of the flux on each constant state.
    - `derivative`: The mean over the interfaces of the derivative of that flux with
        respect to the state. For a consistent flux it equals f'(u).
    - `derivative_spread`: The standard deviation of that derivative over the
        interfaces, as a root mean square over the states.
    - `real_stencil_mismatch`: The relative l2 distance, over the interfaces of the
        last context frame, between the derivative of the flux along a uniform shift
        of that frame and f' at the mean of the two cells next to the interface. It is
        the estimate on real stencils that the constant states are compared with.
    """

    states: Float[Array, " states"]
    flux: Float[Array, " states"]
    derivative: Float[Array, " states"]
    derivative_spread: Float[Array, ""]
    real_stencil_mismatch: Float[Array, ""]


def effective_flux(
    model: HyperFluxFNOLocal,
    context: Float[Array, " time 1 x"],
    coefficients: Float[Array, " params"],
    flux_derivative_fn: FluxFn,
    num_states: int,
) -> EffectiveFlux:
    """Evaluate the flux network that a 1D scalar model generates from a context.

    Entry j of the output of the flux network is the flux through the interface
    between the cells j - 1 and j, and its last entry repeats the first on a periodic
    grid.

    **Arguments:**

    - `model`: The model, already in inference mode.
    - `context`: The context frames.
    - `coefficients`: The coefficients of the true flux of the trajectory.
    - `flux_derivative_fn`: The derivative of the true flux, which only
        `real_stencil_mismatch` uses.
    - `num_states`: The number of constant states.
    """
    if model.num_spatial_dims != 1 or context.shape[1] != 1:
        raise ValueError(
            "The effective flux is defined for one spatial dimension and one channel, "
            f"got a context of shape {context.shape}."
        )
    (flux_model,) = model.generate_flux_models(context)
    num_cells = context.shape[-1]

    def interface_flux(u: Float[Array, " 1 x"]) -> Float[Array, " x"]:
        return flux_model(append_grid_channels(u))[0, :-1]

    def on_constant_state(state: Float[Array, ""]):
        return jax.jvp(
            lambda s: interface_flux(jnp.full((1, num_cells), s)),
            (state,),
            (jnp.ones_like(state),),
        )

    lower, upper = jnp.min(context), jnp.max(context)
    states = lower + (upper - lower) * jnp.linspace(0.0, 1.0, num_states)
    flux, derivative = jax.vmap(on_constant_state)(states)

    u_last = context[-1]
    _, shift_derivative = jax.jvp(interface_flux, (u_last,), (jnp.ones_like(u_last),))
    u_interface = (jnp.roll(u_last[0], 1) + u_last[0]) / 2
    target = flux_derivative_fn(coefficients, u_interface)
    return EffectiveFlux(
        states=states,
        flux=flux.mean(axis=1),
        derivative=derivative.mean(axis=1),
        derivative_spread=jnp.sqrt(jnp.mean(jnp.var(derivative, axis=1))),
        real_stencil_mismatch=jnp.linalg.norm(shift_derivative - target)
        / jnp.linalg.norm(target),
    )


def evaluate_effective_flux(
    model: HyperFluxFNOLocal,
    trajectories: Float[np.ndarray, " trajectories frames 1 x"],
    coefficients: Float[np.ndarray, " trajectories params"],
    window_starts: tuple[int, ...],
    *,
    context_length: int,
    flux_derivative_fn: FluxFn,
    num_states: int,
    chunk_size: int = 1000,
    batch_size: int = 16,
) -> EffectiveFlux:
    """Apply `effective_flux` to the context of every segment of every trajectory.

    **Arguments:**

    - `model`: The model, already in inference mode.
    - `trajectories`: The test trajectories. The array stays on the host.
    - `coefficients`: The coefficients of the true flux of every trajectory.
    - `window_starts`: The frame indices at which the contexts begin.
    - `context_length`: The number of context frames.
    - `flux_derivative_fn`: The derivative of the true flux.
    - `num_states`: The number of constant states.
    - `chunk_size`: The number of trajectories that are moved to the device at once.
    - `batch_size`: The number of trajectories whose contexts run in parallel on the
        device.

    **Returns:**

    An `EffectiveFlux` whose fields are NumPy arrays with the leading shape
    `(trajectories, windows)`.
    """
    _check_window_starts(window_starts, trajectories.shape[1], context_length)
    starts = jnp.asarray(window_starts)

    def one_trajectory(model, u, coefficients_):
        return jax.vmap(
            lambda context: effective_flux(
                model, context, coefficients_, flux_derivative_fn, num_states
            )
        )(_segments_of(u, starts, context_length))

    @eqx.filter_jit
    def evaluate_chunk(model, u_chunk, coefficient_chunk):
        return jax.lax.map(
            lambda xs: one_trajectory(model, *xs),
            (u_chunk, coefficient_chunk),
            batch_size=batch_size,
        )

    chunks = []
    for start in range(0, trajectories.shape[0], chunk_size):
        chunk = slice(start, start + chunk_size)
        result = evaluate_chunk(
            model, jnp.asarray(trajectories[chunk]), jnp.asarray(coefficients[chunk])
        )
        chunks.append(jax.tree.map(np.asarray, result))
    return jax.tree.map(lambda *xs: np.concatenate(xs), *chunks)


def relative_l2_on_grid(
    difference: Float[np.ndarray, " *batch states"],
    reference: Float[np.ndarray, " *batch states"],
) -> Float[np.ndarray, " *batch"]:
    """Return ||difference|| / ||reference|| for two functions on an equally spaced
    grid, with L2 norms by the trapezoidal rule along the last axis.

    The result is not finite where `reference` vanishes on the whole grid.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(
            np.trapezoid(np.square(difference), axis=-1)
            / np.trapezoid(np.square(reference), axis=-1)
        )


def best_cubic_in_box(
    states: Float[np.ndarray, " states"],
    flux_derivative: Float[np.ndarray, " states"],
    bound: float = 1.0,
) -> Float[np.ndarray, " 3"]:
    """Return the cubic flux whose derivative is closest to a given flux derivative,
    among the cubic fluxes with coefficients in `[-bound, bound]`.

    The result is the least-squares fit of 3 a u^2 + 2 b u + c to `flux_derivative`
    at `states`, with every coefficient constrained to the box. With `bound = 1` the
    box is the training family of the 1D cubic models, so the result is the member of
    the training family that explains an unseen flux best over these states.

    **Returns:**

    The coefficients `(a, b, c)`.
    """
    design = np.stack((3 * states**2, 2 * states, np.ones_like(states)), axis=1)
    return lsq_linear(design, flux_derivative, bounds=(-bound, bound)).x


# --- One step of another size (action 7c) --------------------------------------------


class TimeStepErrors(eqx.Module):
    """Errors of single steps of several sizes from one context.

    Every field has one entry per step size.

    **Attributes:**

    - `l2`: The relative l2 error of the model.
    - `l_inf`: The relative l-infinity error of the model.
    - `l2_identity`: The relative l2 error of the identity baseline, which predicts
        the last context frame.
    - `l_inf_identity`: The relative l-infinity error of the identity baseline.
    """

    l2: Float[Array, " steps"]
    l_inf: Float[Array, " steps"]
    l2_identity: Float[Array, " steps"]
    l_inf_identity: Float[Array, " steps"]


def evaluate_time_steps(
    model: HyperFluxFNOLocal,
    trajectories: Float[np.ndarray, " trajectories frames channels *spatial_dims"],
    window_starts: tuple[int, ...],
    model_args: tuple[float, ...],
    *,
    context_length: int,
    frames_per_step: int,
    step_frames: tuple[int, ...],
    chunk_size: int = 1000,
    batch_size: int = 16,
) -> TimeStepErrors:
    """Compute the errors of single steps of several sizes on every segment.

    The trajectories are stored at a finer spacing than the time step of the model:
    `frames_per_step` stored frames make one model step. The context is sampled at
    the time step of the model, and the flux networks are generated from it once.
    They then advance the last context frame by one step that spans `m` stored frames,
    for every `m` in `step_frames`, and the result is compared with the stored frame
    at that time. `m = frames_per_step` is the step that the model was trained on.

    **Arguments:**

    - `model`: The model, already in inference mode.
    - `trajectories`: The test trajectories at the stored spacing. The array stays on
        the host.
    - `window_starts`: The model steps at which the contexts begin. A start `s` is the
        stored frame `s * frames_per_step`.
    - `model_args`: The stored spacing in time, followed by the grid spacing of every
        spatial dimension.
    - `context_length`: The number of context frames.
    - `frames_per_step`: The number of stored frames per time step of the model.
    - `step_frames`: The step sizes, in stored frames.
    - `chunk_size`: The number of trajectories that are moved to the device at once.
    - `batch_size`: The number of trajectories whose segments run in parallel on the
        device.

    **Returns:**

    A `TimeStepErrors` whose fields are NumPy arrays of shape
    `(trajectories, windows, len(step_frames))`.
    """
    if not window_starts:
        raise ValueError("window_starts is empty.")
    last_frames = [
        (start + context_length - 1) * frames_per_step for start in window_starts
    ]
    if (
        min(window_starts) < 0
        or max(last_frames) + max(step_frames) >= (trajectories.shape[1])
    ):
        raise ValueError(
            f"A segment that starts at one of the model steps {window_starts} leaves "
            f"the {trajectories.shape[1]} stored frames."
        )
    dt_stored, *dxs = model_args
    context_frames = (
        jnp.asarray(window_starts)[:, None] + jnp.arange(context_length)
    ) * frames_per_step
    target_frames = jnp.asarray(last_frames)[:, None] + jnp.asarray(step_frames)
    step_sizes = [m * dt_stored for m in step_frames]

    def one_segment(model, context, targets):
        flux_models = model.generate_flux_models(context)
        state = context[-1]
        predictions = jnp.stack(
            [model.step(flux_models, state, (dt, *dxs)) for dt in step_sizes]
        )
        errors = jax.vmap(lambda u_pred, u: segment_errors(u_pred[None], u[None]))(
            predictions, targets
        )
        identity = jax.vmap(lambda u: segment_errors(state[None], u[None]))(targets)
        return TimeStepErrors(
            l2=errors.l2_onestep,
            l_inf=errors.l_inf_onestep,
            l2_identity=identity.l2_onestep,
            l_inf_identity=identity.l_inf_onestep,
        )

    @eqx.filter_jit
    def evaluate_chunk(model, u_chunk):
        return jax.lax.map(
            lambda u: jax.vmap(lambda *xs: one_segment(model, *xs))(
                u[context_frames], u[target_frames]
            ),
            u_chunk,
            batch_size=batch_size,
        )

    return _map_over_chunks(
        lambda u_chunk: evaluate_chunk(model, u_chunk), trajectories, chunk_size
    )
