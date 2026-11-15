"""Generate the graded unseen-flux test sets of the ICLR 2027 rebuttal (item 1b).

One run writes one test set for the checkpoints trained on the 1D cubic family. The
sets are written in the xarray/HDF5 layout of the submitted 1D test sets, with
dimensions (pde, ic, t, dim, x), so that the evaluation notebooks read them unchanged.

Three flux families are available (`flux` in `configs/unseen_flux_1d.yaml`):

- `sine_rescaled`: f(u) = (a'/|b|) sin(b u) with |b| drawn from a bin. The maximum
    wave speed is |a'| for every b, so only the shape of the flux changes with the bin.
- `cubic_shell`: the cubic flux with coefficients outside the training box, binned by
    the largest absolute coefficient.
- `sine_box`: f(u) = a sin(b u) with (a, b) ~ U[-1, 1]^2. With `seed=12` it repeats
    the key derivation of the submitted sine sets, and so solves the same problems with
    the conservative solver.

The sine sets are solved with the Godunov solver of
`context_flux_no.simulations.pde.godunov_1d`. The cubic sets are solved with
`CubicFlux1D.solve`, the solver of the submitted cubic sets.

Every set of one seed uses the same initial conditions, and the sine bins share the
draws of a' and of the sign of b. The error of a model against the bin then varies
only with the flux.

Example usage, the eight sets of the shape axis:

    uv run python scripts/data/generate_unseen_flux_1d.py --multirun \
        flux=sine_rescaled bin_index=0,1,2,3 initial_condition_name=grf,step_min2
"""

import logging
from collections.abc import Callable
from pathlib import Path

import hydra
import jax
import numpy as np
import xarray as xr
from context_flux_no.simulations.pde import CubicFlux1D
from context_flux_no.simulations.pde.godunov_1d import (
    make_scalar_godunov_solver,
    riemann_sine_godunov_1D,
)
from context_flux_no.simulations.pdesolve import pdesolve_pyclaw
from context_flux_no.simulations.utils import sample_coefficients_uniform
from context_flux_no.waveforms.grf import GaussianRandomField1D, PeriodicCov
from context_flux_no.waveforms.step import PeriodicRandomStepFunction1D
from jaxtyping import Array, Float, PRNGKeyArray
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


logger = logging.getLogger(__name__)

# Base key of the test split in the generator that wrote the submitted 1D sets
# (`BASE_KEY_DICT["test"]` in `context_flux_no.simulations.utils`).
TEST_BASE_KEY = 2

# Number of candidate draws for the rejection sampling of `sample_cubic_shell`. A
# candidate is rejected with probability (lower / upper)^3 <= 0.512 for the bins of
# the rebuttal plan, so all 64 are rejected with probability below 1e-18.
NUM_SHELL_CANDIDATES = 64

InitialConditionFn = Callable[
    [Float[np.ndarray, " Nx"], PRNGKeyArray], Float[Array, "1 Nx"]
]
Solution = tuple[
    Float[np.ndarray, "time 1 Nx"],
    Float[np.ndarray, " time"],
    list[Float[np.ndarray, " Nx"]],
]


def make_initial_condition(name: str) -> InitialConditionFn:
    """Return the sampler of an initial-condition family.

    `grf` is the periodic Gaussian random field of Section B.2, with covariance
    exp(-(1 - cos(2 pi r))), which is `PeriodicCov(corr_length=1.0)`. The `grf` entry
    of `configs/initial_condition/` has a different kernel and is not used here.

    `step` is the shock-dominated family of Section B.2: 1 to 5 jumps and plateau
    values in [-1, 1]. A step function with one jump on a periodic domain has a single
    plateau, so about a fifth of this family are constant states. A constant state
    close to zero makes the relative error of a prediction arbitrarily large.

    `step_min2` is the same family with 2 to 5 jumps, which has no constant states.
    """
    match name:
        case "grf":
            return GaussianRandomField1D(PeriodicCov(corr_length=1.0)).sample
        case "step":
            return PeriodicRandomStepFunction1D().sample
        case "step_min2":
            return PeriodicRandomStepFunction1D(num_jumps_min=2).sample
        case _:
            raise ValueError(f"Unknown initial condition: {name!r}")


def sample_sine_rescaled(
    key: PRNGKeyArray, b_bin: tuple[float, float], amplitude_max: float
) -> dict[str, float]:
    """Sample the coefficients of f(u) = a sin(b u) with a = a'/|b|.

    a' ~ U[-amplitude_max, amplitude_max], |b| is uniform on the half-open bin
    (lower, upper], and the sign of b is drawn at random. The three draws use separate
    keys and |b| is an affine map of one uniform draw, so two bins sampled with the
    same key share a', the sign of b and the position of |b| inside the bin.
    """
    lower, upper = b_bin
    key_amplitude, key_b, key_sign = jax.random.split(key, 3)
    amplitude = float(
        jax.random.uniform(key_amplitude, minval=-amplitude_max, maxval=amplitude_max)
    )
    # `jax.random.uniform` draws from [0, 1), which maps to (lower, upper].
    b_abs = upper - (upper - lower) * float(jax.random.uniform(key_b))
    sign = 1.0 if bool(jax.random.bernoulli(key_sign)) else -1.0
    return {"a": amplitude / b_abs, "b": sign * b_abs}


def sample_cubic_shell(
    key: PRNGKeyArray, norm_bin: tuple[float, float]
) -> dict[str, float]:
    """Sample cubic coefficients uniformly from the shell lower < max|c_i| <= upper.

    The draw is uniform on [-upper, upper]^3 with every candidate inside
    [-lower, lower]^3 rejected. The names follow `CubicFlux1D`:
    f(u) = a u^3 + b u^2 + c u.
    """
    lower, upper = norm_bin
    candidates = np.asarray(
        jax.random.uniform(
            key, shape=(NUM_SHELL_CANDIDATES, 3), minval=-upper, maxval=upper
        )
    )
    accepted = candidates[np.max(np.abs(candidates), axis=1) > lower]
    if len(accepted) == 0:
        raise RuntimeError(
            f"All {NUM_SHELL_CANDIDATES} candidates fell inside the box of half-width "
            f"{lower}."
        )
    a, b, c = (float(value) for value in accepted[0])
    return {"a": a, "b": b, "c": c}


def set_name(cfg: DictConfig) -> str:
    """Return the name of the test set that the configuration describes.

    Raises if `bin_index` is missing for a binned family or given for `sine_box`.
    """
    match cfg.flux:
        case "sine_rescaled" | "cubic_shell":
            if cfg.bin_index is None:
                raise ValueError(f"flux={cfg.flux} needs a bin_index.")
            lower, upper = cfg.bins[cfg.flux][cfg.bin_index]
            name = f"{cfg.flux}_{lower:g}-{upper:g}"
            if cfg.flux == "sine_rescaled" and cfg.sine_amplitude_max != 1.0:
                name = f"{name}_amplitude={cfg.sine_amplitude_max:g}"
            return name
        case "sine_box":
            if cfg.bin_index is not None:
                raise ValueError("flux=sine_box has no bins; set bin_index=null.")
            return "sine_box_godunov"
        case _:
            raise ValueError(f"Unknown flux: {cfg.flux!r}")


def sample_coefficients(key: PRNGKeyArray, cfg: DictConfig) -> dict[str, float]:
    match cfg.flux:
        case "sine_rescaled":
            return sample_sine_rescaled(
                key,
                tuple(cfg.bins.sine_rescaled[cfg.bin_index]),
                cfg.sine_amplitude_max,
            )
        case "cubic_shell":
            return sample_cubic_shell(key, tuple(cfg.bins.cubic_shell[cfg.bin_index]))
        case "sine_box":
            return sample_coefficients_uniform(
                key, {"a": (-1.0, 1.0), "b": (-1.0, 1.0)}
            )
        case _:
            raise ValueError(f"Unknown flux: {cfg.flux!r}")


def solve(
    coeffs: dict[str, float],
    initial_condition: Callable[[list[Float[np.ndarray, " Nx"]]], Float[Array, "1 Nx"]],
    cfg: DictConfig,
) -> Solution:
    """Solve one trajectory with PyClaw and return `(u, t, xs)` at all Nt + 1 frames."""
    x_spans, Nxs, t_span = [tuple(cfg.x_span)], [cfg.Nx], tuple(cfg.t_span)
    if cfg.flux == "cubic_shell":
        return CubicFlux1D(**coeffs).solve(
            initial_condition,
            x_spans,
            Nxs,
            t_span,
            cfg.Nt,
            bc="periodic",
            verbose=False,
        )
    return pdesolve_pyclaw(
        make_scalar_godunov_solver(riemann_sine_godunov_1D),
        coeffs,
        initial_condition,
        x_spans,
        Nxs,
        t_span,
        cfg.Nt,
        "periodic",
        verbose=False,
    )


@hydra.main(config_path="./configs", config_name="unseen_flux_1d", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    name = set_name(cfg)
    savepath = (
        Path(hydra.utils.to_absolute_path(cfg.savedir))
        / f"{name}_test_{cfg.initial_condition_name}_seed={cfg.seed}.hdf5"
    )
    if savepath.exists():
        raise FileExistsError(f"A test set already exists at {savepath}.")
    savepath.parent.mkdir(parents=True, exist_ok=True)

    sample_initial_condition = make_initial_condition(cfg.initial_condition_name)
    num_frames = cfg.Nt // cfg.save_every + 1
    if cfg.Nt % cfg.save_every != 0:
        raise ValueError("Nt must be a multiple of save_every.")

    # Key derivation of the generator that wrote the submitted 1D sets: one key per
    # coefficient draw, split into one key for the coefficients and one per initial
    # condition. The keys do not depend on the flux or on the bin.
    base_key = jax.random.key(TEST_BASE_KEY)
    keys = jax.random.split(jax.random.fold_in(base_key, cfg.seed), cfg.n_coeffs)

    values = np.full(
        (cfg.n_coeffs, cfg.n_ics_per_coeff, num_frames, 1, cfg.Nx), np.nan, np.float64
    )
    failed = np.zeros((cfg.n_coeffs, cfg.n_ics_per_coeff), dtype=bool)
    coeffs_all: list[dict[str, float]] = []
    t = x = None

    for i, key in enumerate(tqdm(keys)):
        key_coeff, *keys_ic = jax.random.split(key, cfg.n_ics_per_coeff + 1)
        coeffs = sample_coefficients(key_coeff, cfg)
        coeffs_all.append(coeffs)

        for j, key_ic in enumerate(keys_ic):
            try:
                u, t_all, xs = solve(
                    coeffs,
                    lambda xs_, key_ic=key_ic: sample_initial_condition(xs_[0], key_ic),
                    cfg,
                )
            # TODO: narrow to the exceptions that PyClaw raises on a failed solve.
            except Exception:
                logger.exception("Solve failed for draw %d, ic %d, %s", i, j, coeffs)
                failed[i, j] = True
                continue
            if not np.all(np.isfinite(u)):
                logger.error("Non-finite solution for draw %d, ic %d, %s", i, j, coeffs)
                failed[i, j] = True
                continue
            values[i, j] = u[:: cfg.save_every]
            t, x = t_all[:: cfg.save_every], xs[0]

    if t is None:
        raise RuntimeError("Every solve failed; nothing was written.")
    logger.info("%d of %d solves failed.", failed.sum(), failed.size)

    param_names = list(coeffs_all[0].keys())
    dataset = xr.Dataset(
        {
            "values": (["pde", "ic", "t", "dim", "x"], values),
            "coeffs": (
                ["pde", "param"],
                np.asarray([[c[p] for p in param_names] for c in coeffs_all]),
            ),
            # True where the solve failed; the trajectory is then NaN in `values`.
            "failed": (["pde", "ic"], failed),
        },
        coords={"t": t, "dim": ["u"], "x": x, "param": param_names},
        attrs={
            "set_name": name,
            "flux": cfg.flux,
            "bin": (
                "none"
                if cfg.bin_index is None
                else str(list(cfg.bins[cfg.flux][cfg.bin_index]))
            ),
            "sine_amplitude_max": float(cfg.sine_amplitude_max),
            "initial_condition": cfg.initial_condition_name,
            "seed": int(cfg.seed),
            "Nt_solve": int(cfg.Nt),
            "save_every": int(cfg.save_every),
            "riemann_solver": (
                "riemann_cubic_1D"
                if cfg.flux == "cubic_shell"
                else "riemann_sine_godunov_1D"
            ),
            "num_failed": int(failed.sum()),
        },
    )
    dataset.to_netcdf(savepath, engine="h5netcdf")
    print(f"Wrote {savepath}")


if __name__ == "__main__":
    main()
