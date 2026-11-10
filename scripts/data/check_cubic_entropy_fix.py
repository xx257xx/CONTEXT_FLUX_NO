"""Measure how far the submitted cubic reference solver is from the entropy solution.

`riemann_cubic_1D` moves every jump at its Rankine-Hugoniot speed and has no entropy
fix. A jump that should open into a rarefaction with characteristic speeds of both
signs can then persist as an expansion shock. This script solves shock-dominated
problems of the 1D cubic training family twice, with `riemann_cubic_1D` and with the
Godunov solver `riemann_cubic_godunov_1D`, and prints the relative l2 difference between
the two solutions per trajectory. Nothing is written to disk.

The difference is to be read against the errors of the models on the shock-dominated
cubic test set (relative l2 of about 0.07 over a 20-step rollout for HFluxNO).

Usage:

    uv run python scripts/data/check_cubic_entropy_fix.py
"""

import jax
import numpy as np
from context_flux_no.simulations.pde import CubicFlux1D
from context_flux_no.simulations.pde.godunov_1d import (
    make_scalar_godunov_solver,
    riemann_cubic_godunov_1D,
)
from context_flux_no.simulations.pdesolve import pdesolve_pyclaw
from context_flux_no.simulations.utils import sample_coefficients_uniform
from context_flux_no.waveforms.step import PeriodicRandomStepFunction1D
from jaxtyping import Float
from tqdm import tqdm


NUM_COEFFS = 10
NUM_ICS_PER_COEFF = 10
SEED = 0

# Grid and output times of the submitted 1D cubic sets: Nt = 1000 solver frames on
# t in [0, 0.5], of which every 10th is evaluated.
X_SPANS = [(0.0, 1.0)]
NXS = [100]
T_SPAN = (0.0, 0.5)
NT = 1000
SAVE_EVERY = 10

COEFF_RANGES = {"a": (-1.0, 1.0), "b": (-1.0, 1.0), "c": (-1.0, 1.0)}


def main() -> None:
    sample_initial_condition = PeriodicRandomStepFunction1D().sample
    keys = jax.random.split(jax.random.key(SEED), NUM_COEFFS)

    differences: list[float] = []
    for key in tqdm(keys):
        key_coeff, *keys_ic = jax.random.split(key, NUM_ICS_PER_COEFF + 1)
        coeffs = sample_coefficients_uniform(key_coeff, COEFF_RANGES)
        for key_ic in keys_ic:

            def initial_condition(
                xs: list[Float[np.ndarray, " Nx"]], key_ic=key_ic
            ) -> Float[np.ndarray, "1 Nx"]:
                return np.asarray(sample_initial_condition(xs[0], key_ic))

            u_submitted, _, _ = CubicFlux1D(**coeffs).solve(
                initial_condition,
                X_SPANS,
                NXS,
                T_SPAN,
                NT,
                bc="periodic",
                verbose=False,
            )
            u_godunov, _, _ = pdesolve_pyclaw(
                make_scalar_godunov_solver(riemann_cubic_godunov_1D),
                coeffs,
                initial_condition,
                X_SPANS,
                NXS,
                T_SPAN,
                NT,
                "periodic",
                verbose=False,
            )
            u_submitted = u_submitted[::SAVE_EVERY]
            u_godunov = u_godunov[::SAVE_EVERY]
            differences.append(
                float(
                    np.linalg.norm(u_submitted - u_godunov) / np.linalg.norm(u_godunov)
                )
            )

    differences_array = np.asarray(differences)
    print(f"Relative l2 difference over {len(differences_array)} trajectories")
    for label, value in [
        ("median", np.median(differences_array)),
        ("mean", np.mean(differences_array)),
        ("90th percentile", np.quantile(differences_array, 0.9)),
        ("maximum", np.max(differences_array)),
    ]:
        print(f"  {label:16s} {value:.3e}")
    for threshold in (1e-2, 1e-1):
        fraction = np.mean(differences_array > threshold)
        print(f"  fraction above {threshold:g}: {fraction:.2f}")


if __name__ == "__main__":
    main()
