"""Generate a shock-dominated shallow-water test set from the training family.

This script writes the corrected shock-dominated test set of the 1D shallow-water
family (ICLR 2027 rebuttal plan, item 5). h(0) is a periodic random step function with
plateau values in [0.5, 4.5]. The coefficients and m(0) follow the training set: alpha
and gamma in [0.5, 1.5], beta in [0.8, 1.2], and a Gaussian random field with variance
1 for m(0) (`configs/pde/shallow_water_1d.yaml`, `configs/initial_condition/
grf_shallow.yaml`). The set therefore changes the initial condition of h and nothing
else.

The submitted shock set, `shallow_water_1d_test_step_grf_seed=10.hdf5`, does not have
this property. Cell 12 of `notebooks/data_generation/shallow_water_1D.ipynb` wrote it
with beta in [8, 12] and variance 0.5 for m(0), while every training, validation and
in-distribution test file has beta in [0.8, 1.2] and variance 1. Its gravity waves are
about 3.2 times faster than any that the models saw in training.

The keys are those of the in-distribution test set: the base key of the test split,
seed 10, and the coefficients sampled in the order of the configuration. The 100
coefficient draws of the two sets are then the same.

Every range is a configuration value. The following overrides reproduce the submitted
set, whose first trajectory has the coefficients and the plateau value that cell 13 of
the notebook prints:

    base_key=0 momentum_variance=0.5 set_name=submitted \
    'coeff_ranges=[[gamma,0.5,1.5],[alpha,0.5,1.5],[beta,8,12]]'

The file has the layout of the submitted shallow-water sets, with the dimensions
(ic, t, dim, x), so the evaluation code reads it unchanged. Two variables are added:
`draw_index` and `ic_index` give the position (i, j) of each stored trajectory. A solve
that fails or returns a non-finite value is left out of `values` and listed in
`failed_draw_index` and `failed_ic_index`.

The solver is configured here, as in `ParametrizedShallowWater1D.solve`, and passed to
`pdesolve_pyclaw`. That method cannot be called: it wraps the grid list that
`pdesolve_pyclaw` now returns into a second tuple before building the dataset.

Example usage, the pilot and then the full set:

    uv run python scripts/data/generate_shallow_water_step_1d.py pilot_size=10
    uv run python scripts/data/generate_shallow_water_step_1d.py
"""

import logging
from pathlib import Path

import gstools as gs
import hydra
import jax
import numpy as np
import xarray as xr
from clawpack import pyclaw
from context_flux_no.simulations.pde.shallow_water import (
    ParametrizedShallowWater1D,
    riemann_param_shallow_water_1D,
)
from context_flux_no.simulations.pdesolve import pdesolve_pyclaw
from context_flux_no.simulations.utils import sample_coefficients_uniform
from context_flux_no.waveforms.grf import GaussianRandomField
from context_flux_no.waveforms.step import (
    MultichannelWaveform,
    PeriodicRandomStepFunction1D,
)
from jaxtyping import Float
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


logger = logging.getLogger(__name__)

# The coefficients of `ParametrizedShallowWater1D`, where `gamma` multiplies m in the
# mass flux and `alpha` multiplies m^2 / h, the reverse of the notation of Section B.3.
COEFFICIENT_NAMES = frozenset({"alpha", "gamma", "beta"})


def coefficient_ranges(cfg: DictConfig) -> dict[str, tuple[float, float]]:
    """Return the coefficient ranges of the configuration, in its order.

    The configuration lists the ranges as `[name, lower, upper]` entries. The order
    decides which key samples which coefficient, so it is part of what makes two sets
    share their coefficient draws. A list is used because a command-line override
    replaces a list, while it merges into a mapping and keeps the old order.
    """
    names = [str(entry[0]) for entry in cfg.coeff_ranges]
    if sorted(names) != sorted(COEFFICIENT_NAMES):
        raise ValueError(
            f"coeff_ranges must name each of {sorted(COEFFICIENT_NAMES)} once, "
            f"got {names}."
        )
    return {
        str(name): (float(lower), float(upper))
        for name, lower, upper in cfg.coeff_ranges
    }


def make_initial_condition(
    plateau_range: tuple[float, float],
    momentum_variance: float,
    momentum_length_scale: float,
) -> MultichannelWaveform:
    """Return the sampler of (h(0), m(0)) of a shock-dominated set.

    h(0) is a periodic random step function with plateau values in `plateau_range`.
    m(0) is a Gaussian random field with a Gaussian covariance of the given variance
    and length scale.
    """
    value_min, value_max = plateau_range
    return MultichannelWaveform(
        [
            PeriodicRandomStepFunction1D(value_min=value_min, value_max=value_max),
            GaussianRandomField(
                covariance_fns=[
                    gs.Gaussian(
                        dim=1, var=momentum_variance, len_scale=momentum_length_scale
                    )
                ],
                means=None,
                scales=None,
            ),
        ]
    )


def make_solver() -> pyclaw.ClawSolver1D:
    """Return the PyClaw solver of the submitted shallow-water sets.

    The settings are those of `ParametrizedShallowWater1D.solve`: the Roe solver with
    the MC limiter, target CFL number 0.5 and maximum CFL number 0.9.
    """
    solver = pyclaw.ClawSolver1D(riemann_param_shallow_water_1D)
    solver.limiters = pyclaw.limiters.tvd.MC
    solver.num_eqn = ParametrizedShallowWater1D.n_eqns
    solver.num_waves = 2
    solver.kernel_language = "Python"
    solver.cfl_desired = 0.5
    solver.cfl_max = 0.9
    solver.fwave = False
    return solver


def savepath_of(cfg: DictConfig) -> Path:
    """Return the path of the file that the configuration describes."""
    suffix = "" if cfg.pilot_size is None else f"_pilot{cfg.pilot_size}"
    name = f"shallow_water_1d_test_step_{cfg.set_name}_grf_seed={cfg.seed}{suffix}.hdf5"
    return Path(hydra.utils.to_absolute_path(cfg.savedir)) / name


@hydra.main(
    config_path="./configs", config_name="shallow_water_step_1d", version_base=None
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    savepath = savepath_of(cfg)
    if savepath.exists():
        raise FileExistsError(f"A test set already exists at {savepath}.")
    savepath.parent.mkdir(parents=True, exist_ok=True)

    num_draws = cfg.n_coeffs if cfg.pilot_size is None else cfg.pilot_size
    num_ics = cfg.n_ics_per_coeff if cfg.pilot_size is None else cfg.pilot_size
    if num_draws > cfg.n_coeffs or num_ics > cfg.n_ics_per_coeff:
        raise ValueError(
            f"pilot_size={cfg.pilot_size} exceeds n_coeffs={cfg.n_coeffs} or "
            f"n_ics_per_coeff={cfg.n_ics_per_coeff}."
        )

    initial_condition = make_initial_condition(
        tuple(cfg.plateau_range), cfg.momentum_variance, cfg.momentum_length_scale
    )
    coeff_ranges = coefficient_ranges(cfg)

    # One key per coefficient draw, split into one key for the coefficients and one
    # per initial condition. The keys are split at the full sizes and then truncated,
    # so a pilot solves trajectories of the full set.
    base_key = jax.random.key(cfg.base_key)
    keys = jax.random.split(jax.random.fold_in(base_key, cfg.seed), cfg.n_coeffs)

    values: list[Float[np.ndarray, " time 2 Nx"]] = []
    coeffs_all: list[list[float]] = []
    kept: list[tuple[int, int]] = []
    failed: list[tuple[int, int]] = []
    param_names: list[str] = []
    t = x = None

    for i, key in enumerate(tqdm(keys[:num_draws])):
        key_coeff, *keys_ic = jax.random.split(key, cfg.n_ics_per_coeff + 1)
        pde = ParametrizedShallowWater1D(
            **sample_coefficients_uniform(key_coeff, coeff_ranges)
        )
        param_names = list(pde.coeffs.keys())

        for j, key_ic in enumerate(keys_ic[:num_ics]):
            try:
                u, t_all, xs = pdesolve_pyclaw(
                    make_solver(),
                    pde.coeffs,
                    lambda xs_, key_ic=key_ic: initial_condition.sample(xs_[0], key_ic),
                    [tuple(cfg.x_span)],
                    [cfg.Nx],
                    tuple(cfg.t_span),
                    cfg.Nt,
                    "periodic",
                    verbose=False,
                )
            # TODO: narrow to the exceptions that PyClaw raises on a failed solve.
            except Exception:
                logger.exception("Solve failed for draw %d, ic %d", i, j)
                failed.append((i, j))
                continue
            if not np.all(np.isfinite(u)):
                logger.error("Non-finite solution for draw %d, ic %d", i, j)
                failed.append((i, j))
                continue
            values.append(u)
            coeffs_all.append(list(pde.coeffs.values()))
            kept.append((i, j))
            t, x = t_all, xs[0]

    if t is None:
        raise RuntimeError("Every solve failed; nothing was written.")
    logger.info("%d of %d solves failed.", len(failed), num_draws * num_ics)

    kept_array = np.asarray(kept, dtype=np.int64)
    failed_array = np.asarray(failed, dtype=np.int64).reshape(-1, 2)
    dataset = xr.Dataset(
        {
            "values": (["ic", "t", "dim", "x"], np.stack(values)),
            "coeffs": (["ic", "param"], np.asarray(coeffs_all)),
            "draw_index": (["ic"], kept_array[:, 0]),
            "ic_index": (["ic"], kept_array[:, 1]),
            "failed_draw_index": (["failed"], failed_array[:, 0]),
            "failed_ic_index": (["failed"], failed_array[:, 1]),
        },
        coords={"t": t, "dim": ["u_0", "u_1"], "x": x, "param": param_names},
        attrs={
            "plateau_range": [float(v) for v in cfg.plateau_range],
            "momentum_variance": float(cfg.momentum_variance),
            "momentum_length_scale": float(cfg.momentum_length_scale),
            "coeff_ranges": str(coeff_ranges),
            "seed": int(cfg.seed),
            "base_key": int(cfg.base_key),
            "n_coeffs": int(cfg.n_coeffs),
            "n_ics_per_coeff": int(cfg.n_ics_per_coeff),
            "pilot_size": -1 if cfg.pilot_size is None else int(cfg.pilot_size),
            "riemann_solver": "riemann_param_shallow_water_1D",
            "num_failed": len(failed),
        },
    )
    dataset.to_netcdf(savepath, engine="h5netcdf")
    print(f"Wrote {savepath}")


if __name__ == "__main__":
    main()
