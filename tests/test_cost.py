"""Checks of the cost measurements in `context_flux_no.cost`.

The FLOP count is compared with the closed form of a matrix product, and the parameter
breakdown with a flux network built directly from the settings of the small HFluxNO.
"""

import jax
import jax.numpy as jnp
import numpy as np
from context_flux_no.cost import count_flops, measure, parameter_breakdown
from context_flux_no.models.multiphysics.hyperfluxfno.hyperfluxfno import FluxModel
from context_flux_no.utils import num_parameters
from helpers import make_small_hfluxno
from jaxtyping import Array, Float


def _matmul(a: Float[Array, "m k"], b: Float[Array, "k n"]) -> Float[Array, "m n"]:
    return a @ b


def test_count_flops_of_a_matrix_product():
    a, b = jnp.ones((6, 5)), jnp.ones((5, 4))
    assert count_flops(_matmul, a, b) == 2 * 6 * 5 * 4


def test_measure_times_every_repeat():
    a, b = jnp.ones((6, 5)), jnp.ones((5, 4))
    measurement = measure(_matmul, a, b, repeats=3, warmup=1)
    assert measurement.seconds.shape == (3,)
    assert np.all(np.isfinite(measurement.seconds) & (measurement.seconds > 0))
    assert measurement.flops == 2 * 6 * 5 * 4
    assert measurement.output_bytes == 6 * 4 * 4


def test_parameter_breakdown_of_hfluxno():
    model = make_small_hfluxno(num_cells=16)
    # The flux network of `make_small_hfluxno`, built without the hypernetwork. Its
    # input is the state and one grid channel.
    flux_model = FluxModel(
        in_channels=2,
        out_channels=1,
        stencil_widths=(2, 2),
        lift_dim=8,
        hidden_dim=8,
        depth=1,
        key=jax.random.key(1),
    )
    counts = parameter_breakdown(model)
    assert counts["total"] == num_parameters(model)
    assert counts["other"] == 0
    assert counts["generated"] == num_parameters(flux_model)
