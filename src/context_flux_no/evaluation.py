"""Rollout evaluation of trained models on a set of test trajectories.

This module computes the rollout diagnostics of `context_flux_no.metrics` (D1 to D6 of
the ICLR 2027 rebuttal plan, action 2b). `evaluate_rollouts` and `evaluate_references`
run on the accelerator, and `summarize_rollouts` reduces their per-step arrays on the
host. `scripts/evaluation/rollout_diagnostics.py` applies them to the test sets of the
paper.
"""

import dataclasses
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from .metrics import (
    error_power_spectrum,
    first_true_step,
    ideal_gas_pressure,
    rollout_diagnostics,
    rollout_references,
    RolloutDiagnostics,
    RolloutReferences,
    valid_horizon,
)
from .models.multiphysics.abstract import AbstractMultiphysicsOperator


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

    diagnostics = rollout_diagnostics(u_pred, u_data, area_thresholds)
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
    minima: Float[np.ndarray, " *batch trajectories steps quantities"] | None = None,
    horizon: int = 80,
    report_steps: tuple[int, ...] = (20, 40, 80),
) -> dict[str, Any]:
    """Reduce the per-step arrays of one model to the numbers of D2 to D5.

    The leading `*batch` axes, such as the seeds of one model, are pooled with the
    trajectories. Only the first `horizon` steps are used.

    **Arguments:**

    - `l2`, `nonfinite`, `area_fraction`, `total_variation`: The fields of
        `RolloutDiagnostics`.
    - `references`: The reference quantities of the same trajectories.
    - `minima`: The minima of the quantities that must stay positive, or `None`.
    - `horizon`: The number of rollout steps that enter the summary.
    - `report_steps`: The steps at which the failure fractions, the area fraction and
        the total-variation ratio are reported.

    **Returns:**

    A dictionary with the following entries.

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

    l2 = l2[..., :horizon]
    nonfinite = nonfinite[..., :horizon]
    l2_identity = references.l2_identity[..., :horizon]
    l2_spatial_mean = references.l2_spatial_mean[..., :horizon]

    horizons = valid_horizon(l2, l2_identity, nonfinite)
    with np.errstate(invalid="ignore"):
        diverged = (l2 > l2_spatial_mean) | nonfinite | ~np.isfinite(l2)
        ratio = (
            total_variation[..., :horizon] / references.total_variation[..., :horizon]
        )

    summary = {
        "num_rollouts": int(horizons.size),
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
