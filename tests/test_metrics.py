"""Analytic checks of the rollout diagnostics in `context_flux_no.metrics`.

Each test compares one definition with a value that is known in closed form.
"""

import jax
import jax.numpy as jnp
import numpy as np
from context_flux_no.metrics import (
    count_jumps,
    error_area_fraction,
    error_power_spectrum,
    first_true_step,
    ideal_gas_pressure,
    largest_jump,
    radial_power_spectrum,
    relative_L2_error,
    relative_L2_error_per_channel,
    rollout_diagnostics,
    rollout_references,
    segment_errors,
    total_variation,
    valid_horizon,
)


# Agreement of a closed form with its float32 evaluation, for sums over at most
# 64 x 64 cells of order-one values.
ATOL = 1e-4
RTOL = 1e-4


def _grid(n: int) -> jnp.ndarray:
    return jnp.linspace(0.0, 1.0, n, endpoint=False)


def test_total_variation_of_a_sine():
    # A sine of amplitude A with m periods rises and falls by 2A per half period. The
    # grid of 64 cells samples the extrema of m = 2 exactly.
    amplitude, periods = 1.5, 2
    u = amplitude * jnp.sin(2 * jnp.pi * periods * _grid(64))
    assert jnp.allclose(total_variation(u), 4 * amplitude * periods, atol=ATOL)


def test_total_variation_sums_over_axes():
    # u(x, y) = f(x) varies along x only, so every one of the 16 rows along y
    # contributes the 1D total variation once.
    f = jnp.sin(2 * jnp.pi * _grid(64))
    u = jnp.broadcast_to(f[:, None], (64, 16))
    assert jnp.allclose(total_variation(u), 16 * 4.0, atol=ATOL)


def test_error_area_fraction_counts_cells_above_threshold():
    # The reference has range 2. The error is 0.3 on 5 of 100 cells, which exceeds
    # 0.1 * 2 and not 0.2 * 2.
    u_data = jnp.linspace(-1.0, 1.0, 100)
    u_pred = u_data.at[10:15].add(0.3)
    fractions = error_area_fraction(u_pred, u_data, jnp.array([0.05, 0.1, 0.2]))
    assert jnp.allclose(fractions, jnp.array([0.05, 0.05, 0.0]), atol=ATOL)


def test_radial_power_spectrum_of_single_modes():
    # cos(2 pi (3x + 4y)) has wavenumber magnitude 5 and mean square 1/2. The constant
    # 2 lies in bin 0 with mean square 4.
    x, y = jnp.meshgrid(_grid(32), _grid(32), indexing="ij")
    u = 2.0 + jnp.cos(2 * jnp.pi * (3 * x + 4 * y))
    spectrum = radial_power_spectrum(u)
    expected = jnp.zeros_like(spectrum).at[0].set(4.0 * u.size).at[5].set(0.5 * u.size)
    assert jnp.allclose(spectrum, expected, rtol=RTOL, atol=ATOL * u.size)


def test_error_power_spectrum_sums_to_squared_relative_error():
    key_pred, key_data = jax.random.split(jax.random.key(0))
    u_data = 1.0 + jax.random.normal(key_data, (32, 48))
    u_pred = u_data + 0.1 * jax.random.normal(key_pred, (32, 48))
    spectrum = error_power_spectrum(u_pred, u_data)
    expected = relative_L2_error(u_pred[None], u_data[None]) ** 2
    assert jnp.allclose(jnp.sum(spectrum), expected, rtol=RTOL)


def test_relative_l2_error_per_channel():
    u_data = jnp.stack((jnp.ones(50), 10.0 * jnp.ones(50)))
    u_pred = u_data + jnp.array([[0.1], [0.0]])
    errors = relative_L2_error_per_channel(u_pred, u_data)
    assert jnp.allclose(errors, jnp.array([0.1, 0.0]), atol=ATOL)


def test_ideal_gas_pressure_inverts_the_energy():
    # E = p / (gamma - 1) + rho |v|^2 / 2, the relation of simulations.pde.euler.
    density, pressure, gamma = jnp.full((4, 4), 1.3), jnp.full((4, 4), 0.7), 1.4
    velocity = jnp.stack((jnp.full((4, 4), 0.5), jnp.full((4, 4), -2.0)))
    energy = pressure / (gamma - 1) + 0.5 * density * jnp.sum(velocity**2, axis=0)
    recovered = ideal_gas_pressure(density, density * velocity, energy, gamma)
    assert jnp.allclose(recovered, pressure, atol=ATOL)


def test_rollout_references_of_a_translating_wave():
    # u(x, t) = 1 + sin(2 pi (x - t)) has mean 1 and fluctuation power 1/2, so the
    # spatial-mean error is sqrt((1/2) / (3/2)). After a shift of half a period the
    # wave is 1 - sin, so the identity-baseline error is sqrt(2 / (3/2)).
    x = _grid(64)
    shifts = jnp.array([0.0, 0.25, 0.5])
    u = (1.0 + jnp.sin(2 * jnp.pi * (x - shifts[:, None])))[:, None]
    references = rollout_references(u[0], u[1:])
    assert jnp.allclose(references.l2_spatial_mean, jnp.sqrt(1 / 3), atol=ATOL)
    assert jnp.allclose(references.l2_identity[1], jnp.sqrt(4 / 3), atol=ATOL)
    assert jnp.allclose(references.total_variation, 4.0, atol=ATOL)


def test_rollout_diagnostics_flags_nonfinite_steps():
    u_data = jnp.ones((3, 2, 16)) + jnp.sin(2 * jnp.pi * _grid(16))
    u_pred = u_data.at[1, 1, 3].set(jnp.nan)
    diagnostics = rollout_diagnostics(u_pred, u_data, jnp.array([0.1]))
    assert diagnostics.nonfinite.tolist() == [False, True, False]
    assert jnp.allclose(diagnostics.l2[0], 0.0, atol=ATOL)
    assert diagnostics.area_fraction.shape == (3, 1)
    assert diagnostics.l2_per_channel.shape == (3, 2)


def test_first_true_step_counts_from_one():
    flags = np.array([[False, True, True], [False, False, False], [True, False, False]])
    assert first_true_step(flags).tolist() == [2, 4, 1]


def test_valid_horizon():
    l2_identity = np.array([0.1, 0.2, 0.3, 0.4])
    l2 = np.array(
        [
            [0.05, 0.1, 0.3, 0.2],  # reaches the baseline at step 3
            [0.2, 0.1, 0.1, 0.1],  # above the baseline at step 1 only
            [0.0, 0.1, 0.2, 0.3],  # never reaches it
            [0.0, np.nan, np.nan, np.nan],  # non-finite from step 2
        ]
    )
    nonfinite = np.isnan(l2)
    assert valid_horizon(l2, l2_identity, nonfinite).tolist() == [3, 1, 5, 2]


def test_segment_errors_of_a_constant_offset():
    # The reference is 1 in the first frame and 2 in the second, and the prediction is
    # off by 0.5 in the first frame and by 1 in the second. The single-step errors are
    # 0.5 / 1. The rollout l2 error is sqrt(0.25 + 1) / sqrt(1 + 4) = 0.5, and the
    # rollout l-infinity error is 1 / 2.
    u_data = jnp.stack((jnp.ones((2, 8)), 2 * jnp.ones((2, 8))))
    u_pred = u_data + jnp.asarray([0.5, 1.0])[:, None, None]
    errors = segment_errors(u_pred, u_data)
    assert jnp.allclose(errors.l2_onestep, 0.5, atol=ATOL)
    assert jnp.allclose(errors.l_inf_onestep, 0.5, atol=ATOL)
    assert jnp.allclose(errors.l2_rollout, 0.5, atol=ATOL)
    assert jnp.allclose(errors.l_inf_rollout, 0.5, atol=ATOL)


def test_jumps_of_a_step_function():
    # Three plateaus on a periodic grid have three discontinuities, the one across
    # the boundary included, and the largest is the one from 4 down to 0.5.
    u = jnp.asarray([0.5, 0.5, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0])
    assert int(count_jumps(u)) == 3
    assert jnp.allclose(largest_jump(u), 3.5, atol=ATOL)
    assert int(count_jumps(jnp.full(8, 1.5))) == 0
