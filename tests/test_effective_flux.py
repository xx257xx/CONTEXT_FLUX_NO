"""Checks of the effective flux and of the steps of another size in
`context_flux_no.evaluation`.

The effective flux is compared with a flux network that is written by hand from a
known flux, and its derivative with a finite difference of its values. The steps of
another size are compared with the segment evaluation on subsampled trajectories.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from context_flux_no.evaluation import (
    best_cubic_in_box,
    cubic_flux,
    cubic_flux_derivative,
    effective_flux,
    evaluate_effective_flux,
    evaluate_segment_baselines,
    evaluate_segments,
    evaluate_time_steps,
    relative_l2_on_grid,
    sine_flux,
    sine_flux_derivative,
)
from context_flux_no.models.multiphysics import HyperFluxFNOLocal
from context_flux_no.models.multiphysics.hyperfluxfno.hyperfluxfno import FluxModel
from helpers import make_small_hfluxno
from jaxtyping import Array, Float


# Agreement of a float32 evaluation of a polynomial of order-one values with its
# closed form.
ATOL = 1e-5
# Agreement of two float32 evaluations of the same sequence of operations in another
# grouping, for order-one states.
SAME_OPERATIONS_ATOL = 1e-5
# Agreement of a derivative with the central difference of float32 values on 33 states
# over an interval of length 2: the truncation error is (2 / 32)**2 / 6 times the
# third derivative, and the rounding error is about 1e-6 / (2 / 32).
FINITE_DIFFERENCE_ATOL = 2e-2
# The spacing of the grid search that the constrained fit is compared with.
GRID_SEARCH_SPACING = 0.1

NUM_CELLS = 16
CONTEXT_LENGTH = 5
COEFFICIENTS = jnp.asarray([0.4, -0.7, 0.3])


class CentralFluxModel(eqx.Module):
    """A flux network written by hand: the cubic flux at the mean of the two cells
    next to an interface. It is consistent, so on a constant state it returns the flux
    of that state at every interface."""

    coefficients: Float[Array, " 3"]

    def __call__(self, v: Float[Array, " 2 x"]) -> Float[Array, " 1 x+1"]:
        u = v[0]
        at_interface = cubic_flux(self.coefficients, (jnp.roll(u, 1) + u) / 2)
        return jnp.concatenate((at_interface, at_interface[:1]))[None]


class CentralFluxOperator(eqx.Module):
    """A stand-in for a model that generates `CentralFluxModel` from every context."""

    coefficients: Float[Array, " 3"]
    num_spatial_dims: int = 1

    def generate_solver(self, context: Any) -> tuple[CentralFluxModel]:
        return (CentralFluxModel(self.coefficients),)


def _context() -> Float[Array, " time 1 x"]:
    x = jnp.arange(NUM_CELLS) / NUM_CELLS
    t = jnp.arange(CONTEXT_LENGTH)[:, None] * 0.03
    return (jnp.sin(2 * jnp.pi * (x - t)) + 0.2)[:, None, :]


@pytest.fixture(scope="module")
def model() -> HyperFluxFNOLocal:
    return make_small_hfluxno(NUM_CELLS)


def test_effective_flux_of_a_consistent_flux_is_the_flux():
    context = _context()
    result = effective_flux(
        CentralFluxOperator(COEFFICIENTS),
        context,
        COEFFICIENTS,
        cubic_flux_derivative,
        9,
    )
    assert np.allclose(result.states[0], context.min(), atol=ATOL)
    assert np.allclose(result.states[-1], context.max(), atol=ATOL)
    assert np.allclose(result.flux, cubic_flux(COEFFICIENTS, result.states), atol=ATOL)
    assert np.allclose(
        result.derivative, cubic_flux_derivative(COEFFICIENTS, result.states), atol=ATOL
    )
    # The flux does not read the grid coordinate, and on real stencils its derivative
    # along a uniform shift is f' at the interface value.
    assert float(result.derivative_spread) <= ATOL
    assert float(result.real_stencil_mismatch) <= ATOL


def test_real_stencil_mismatch_sees_another_flux():
    other = jnp.asarray([0.4, -0.7, 0.9])
    result = effective_flux(
        CentralFluxOperator(COEFFICIENTS), _context(), other, cubic_flux_derivative, 9
    )
    # The derivatives differ by the constant 0.6 at every interface.
    assert float(result.real_stencil_mismatch) > 0.1


def test_flux_network_output_is_the_interface_left_of_its_cell():
    # `effective_flux` takes entry j of the flux network as the interface between the
    # cells j - 1 and j. With a stencil of 2 cells to each side, that interface reads
    # the cells j - 3 to j + 1, so a change of cell k reaches the entries k - 1 to
    # k + 3 and no other.
    flux_model = FluxModel(
        in_channels=1,
        out_channels=1,
        stencil_widths=(2, 2),
        lift_dim=4,
        hidden_dim=4,
        depth=1,
        key=jax.random.key(1),
    )
    u = jnp.sin(2 * jnp.pi * jnp.arange(NUM_CELLS) / NUM_CELLS)[None]
    k = 8
    changed = flux_model(u.at[0, k].add(0.5))[0] != flux_model(u)[0]
    assert np.flatnonzero(changed).tolist() == [k - 1, k, k + 1, k + 2, k + 3]


def test_effective_flux_derivative_is_the_slope_of_the_effective_flux(model):
    result = effective_flux(model, _context(), COEFFICIENTS, cubic_flux_derivative, 33)
    slope = np.gradient(np.asarray(result.flux), np.asarray(result.states))
    assert np.allclose(
        result.derivative[1:-1], slope[1:-1], atol=FINITE_DIFFERENCE_ATOL
    )
    # The flux network of the model reads the grid coordinate.
    assert float(result.derivative_spread) > 0


def test_effective_flux_of_every_segment(model):
    trajectories = np.stack((_context(), 0.5 * _context()))
    trajectories = np.concatenate((trajectories, trajectories), axis=1)
    coefficients = np.stack((COEFFICIENTS, COEFFICIENTS))
    window_starts = (0, 3)
    result = evaluate_effective_flux(
        model,
        trajectories,
        coefficients,
        window_starts,
        context_length=CONTEXT_LENGTH,
        flux_derivative_fn=cubic_flux_derivative,
        num_states=9,
        chunk_size=1,
        batch_size=1,
    )
    assert result.derivative.shape == (2, len(window_starts), 9)
    assert result.real_stencil_mismatch.shape == (2, len(window_starts))
    single = effective_flux(
        model,
        jnp.asarray(trajectories[1, 3 : 3 + CONTEXT_LENGTH]),
        COEFFICIENTS,
        cubic_flux_derivative,
        9,
    )
    assert np.allclose(
        result.derivative[1, 1], single.derivative, atol=SAME_OPERATIONS_ATOL
    )
    assert np.allclose(result.states[1, 1], single.states, atol=SAME_OPERATIONS_ATOL)


def test_flux_families_have_matching_derivatives():
    u = jnp.linspace(-1.5, 1.5, 7)
    for flux, derivative, coefficients in (
        (cubic_flux, cubic_flux_derivative, COEFFICIENTS),
        (sine_flux, sine_flux_derivative, jnp.asarray([0.5, 2.5])),
    ):
        expected = jax.vmap(jax.grad(lambda s: flux(coefficients, s)))(u)
        assert np.allclose(derivative(coefficients, u), expected, atol=ATOL)
    # f(u) = 0.5 sin(2.5 u) at u = 1 and f(u) = u^3 - u at u = 2.
    assert np.isclose(sine_flux(jnp.asarray([0.5, 2.5]), 1.0), 0.5 * np.sin(2.5))
    assert np.isclose(cubic_flux(jnp.asarray([1.0, 0.0, -1.0]), 2.0), 6.0)


def test_relative_l2_on_grid_of_closed_forms():
    states = np.linspace(0.0, 1.0, 129)
    # ||u|| / ||1|| on [0, 1] is sqrt(1 / 3).
    assert relative_l2_on_grid(states, np.ones_like(states)) == pytest.approx(
        np.sqrt(1 / 3), abs=1e-4
    )
    assert relative_l2_on_grid(np.zeros(5), np.ones(5)) == 0.0
    assert not np.isfinite(relative_l2_on_grid(np.ones(5), np.zeros(5)))


def test_best_cubic_in_box_recovers_a_member_of_the_box():
    states = np.linspace(-1.0, 2.0, 33)
    coefficients = np.asarray([0.4, -0.7, 0.3])
    derivative = np.asarray(cubic_flux_derivative(jnp.asarray(coefficients), states))
    assert np.allclose(best_cubic_in_box(states, derivative), coefficients, atol=1e-4)


@pytest.mark.parametrize(
    "target",
    [
        lambda u: 2.0 + 0 * u,
        lambda u: np.asarray(sine_flux_derivative(jnp.asarray([1 / 3, 3.0]), u)),
        lambda u: 6 * u**2 - 3 * u + 0.5,
    ],
)
def test_best_cubic_in_box_matches_a_grid_search(target):
    states = np.linspace(-1.0, 1.0, 33)
    derivative = target(states)
    fitted = best_cubic_in_box(states, derivative)
    assert np.all(np.abs(fitted) <= 1.0 + 1e-9)

    def residual(coefficients):
        a, b, c = np.moveaxis(np.asarray(coefficients), -1, 0)
        fit = 3 * a[..., None] * states**2 + 2 * b[..., None] * states + c[..., None]
        return np.sum((fit - derivative) ** 2, axis=-1)

    axis = np.arange(-1.0, 1.0 + GRID_SEARCH_SPACING / 2, GRID_SEARCH_SPACING)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    # The fit is the minimum over the box, so no point of the grid has a smaller
    # residual, and the best point of the grid is close to it.
    assert residual(fitted) <= residual(grid).min() + 1e-9
    best = grid.reshape(-1, 3)[residual(grid).argmin()]
    assert np.all(np.abs(fitted - best) <= GRID_SEARCH_SPACING)


def _fine_trajectories() -> Float[np.ndarray, " 2 frames 1 x"]:
    x = np.arange(NUM_CELLS) / NUM_CELLS
    t = np.arange(41)[:, None] * 0.01
    waves = [np.sin(2 * np.pi * (x - t)), 0.5 * np.cos(4 * np.pi * (x + t)) + 0.1]
    return np.stack(waves)[:, :, None, :].astype(np.float32)


def test_the_training_step_is_the_single_step_of_the_segment_evaluation(model):
    fine = _fine_trajectories()
    window_starts = (0, 3, 9)
    dt_stored, dx = 0.0025, 1 / NUM_CELLS
    errors = evaluate_time_steps(
        model,
        fine,
        window_starts,
        (dt_stored, dx),
        context_length=CONTEXT_LENGTH,
        frames_per_step=2,
        step_frames=(1, 2, 4),
        batch_size=1,
    )
    assert errors.l2.shape == (2, len(window_starts), 3)
    # At two stored frames per step, every second frame is a trajectory at the time
    # step of the model, on which the segment evaluation gives the same step.
    kwargs = dict(context_length=CONTEXT_LENGTH, rollout_steps=1)
    segments = evaluate_segments(
        model, fine[:, ::2], window_starts, (2 * dt_stored, dx), batch_size=1, **kwargs
    )
    baselines = evaluate_segment_baselines(fine[:, ::2], window_starts, **kwargs)
    assert np.allclose(
        errors.l2[..., 1], segments.l2_onestep, atol=SAME_OPERATIONS_ATOL
    )
    assert np.allclose(
        errors.l_inf[..., 1], segments.l_inf_onestep, atol=SAME_OPERATIONS_ATOL
    )
    assert np.allclose(
        errors.l2_identity[..., 1], baselines.identity.l2_onestep, atol=ATOL
    )


def test_the_identity_baseline_of_a_step_spans_its_stored_frames(model):
    # Frame n holds the constant n + 1. With two stored frames per step and a context
    # of 5 frames from step 0, the last context frame is stored frame 8 with the value
    # 9, and a step over m stored frames is compared with the value 9 + m.
    frames = np.arange(1, 42, dtype=np.float32)
    trajectories = np.broadcast_to(frames[:, None, None], (1, 41, 1, NUM_CELLS)).copy()
    errors = evaluate_time_steps(
        model,
        trajectories,
        (0,),
        (0.0025, 1 / NUM_CELLS),
        context_length=CONTEXT_LENGTH,
        frames_per_step=2,
        step_frames=(1, 2, 4),
    )
    expected = [m / (9 + m) for m in (1, 2, 4)]
    assert np.allclose(errors.l2_identity[0, 0], expected, atol=ATOL)


def test_a_step_is_linear_in_the_time_step(model):
    context = _context()
    flux_models = model.generate_solver(context)
    dx = 1 / NUM_CELLS
    one = model.step(flux_models, context[-1], (0.005, dx)) - context[-1]
    two = model.step(flux_models, context[-1], (0.01, dx)) - context[-1]
    assert np.allclose(two, 2 * one, atol=SAME_OPERATIONS_ATOL)


def test_a_segment_past_the_last_stored_frame_is_rejected(model):
    with pytest.raises(ValueError, match="stored frames"):
        evaluate_time_steps(
            model,
            _fine_trajectories(),
            (16,),
            (0.0025, 1 / NUM_CELLS),
            context_length=CONTEXT_LENGTH,
            frames_per_step=2,
            step_frames=(1,),
        )
