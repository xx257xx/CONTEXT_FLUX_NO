"""The flux that HFluxNO generates from a context, compared with the true flux.

This script tests whether the context identifies the flux (ICLR 2027 rebuttal plan,
action 3b). A consistent numerical flux G satisfies G(u, ..., u) = f(u) + c on every
constant state u. On every context of a 1D test set the script evaluates the generated
flux network on constant states between the smallest and the largest value of the
context, and compares the result with the true flux of the trajectory.

    python scripts/evaluation/effective_flux.py --preset cubic_1d
    python scripts/evaluation/effective_flux.py --preset sine_rescaled_2-3_grf

Every value of a preset is also a command-line argument, and an argument overrides the
preset. `--limit N` restricts a run to the first N trajectories and writes to a
separate file. A run with `--limit 10` on `cubic_1d` is the pilot that decides whether
constant states can be used: it compares `derivative` with `real_stencil` below.

The script writes the effective flux of every context to one HDF5 file and prints
these relative L2 mismatches over the states of a context:

- `derivative`: the derivative of the effective flux against f'. It has no additive
    constant, and f' is the wave speed.
- `flux`: the effective flux against f, each after its mean over the states is removed.
- `spread`: the standard deviation of the derivative over the interfaces, relative to
    the norm of f'. The flux network reads the grid coordinate, so it is not zero.
- `real_stencil`: the same derivative taken on the last context frame, not on
    constant states, against f' at the interface values.

For a flux outside the cubic family the script also fits the best cubic of the
training box to f' on the states of every context, and prints

- `box_cubic`: that cubic against f', which is the error of the best member of the
    training family,
- `to_box_cubic`: the effective flux against that cubic,
- the fraction of contexts on which the effective flux is closer to f' than that cubic.

The loader and the checkpoint resolution are those of `rollout_diagnostics.py`, and
the coefficients are read as in `context_length.py`. Both are in this directory,
which Python finds because it puts the directory of a script on the path.
"""

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from context_flux_no.evaluation import (
    best_cubic_in_box,
    cubic_flux_derivative,
    EffectiveFlux,
    evaluate_effective_flux,
    FLUX_FAMILIES,
    relative_l2_on_grid,
)
from context_flux_no.training.io import load_model
from context_length import CUBIC_COEFFICIENT_NAMES, load_coefficients
from jaxtyping import Float
from rollout_diagnostics import load_legacy_hdf5, LoadedTestSet, resolve_checkpoints


_HFLUXNO = "cubicflux_1d/HyperFluxFNOLocal/OneStepLoss"
_TEST_DIR = "datasets/cubic_no_source/data/test"
_SUBMITTED = {
    "checkpoint": [
        f"HyperFluxFNOLocal={_HFLUXNO}/seed=0/26-04-16-00_31_47",
        f"HyperFluxFNOLocal={_HFLUXNO}/seed=10/26-04-17-07_57_49",
        f"HyperFluxFNOLocal={_HFLUXNO}/seed=20/26-04-17-10_50_08",
    ],
    "window_starts": list(range(2, 60, 6)),
}
# Data paths are relative to --data-root and checkpoints to --checkpoint-root. The
# window starts are those of the unseen-flux tests (action 1b).
PRESETS: dict[str, dict[str, Any]] = {
    "cubic_1d": {
        "data_path": f"{_TEST_DIR}/cubic_no_source_large_test_seed=10.hdf5",
        "time_stride": 10,
        **_SUBMITTED,
    },
    **{
        f"sine_rescaled_{b_bin}_grf": {
            "data_path": f"{_TEST_DIR}/sine_rescaled_{b_bin}_test_grf_seed=13.hdf5",
            "time_stride": 1,
            **_SUBMITTED,
        }
        for b_bin in ("0-1", "1-2", "2-3", "3-4")
    },
}

DEFAULTS: dict[str, Any] = {
    "data_root": "./data",
    "checkpoint_root": "./checkpoints",
    "output_dir": "./data/analysis/effective_flux",
    "time_stride": 1,
    "context_length": 20,
    "num_states": 65,
    "limit": None,
    "chunk_size": 2000,
    "batch_size": 50,
    "gpu": None,
}


def true_flux_on_states(
    flux_fn, coefficients: Float[np.ndarray, " trajectories params"], states: np.ndarray
) -> Float[np.ndarray, " trajectories windows states"]:
    """Evaluate a flux or its derivative of every trajectory on its states."""
    return np.asarray(jax.vmap(flux_fn)(jnp.asarray(coefficients), jnp.asarray(states)))


def fit_box_cubics(
    states: Float[np.ndarray, " trajectories windows states"],
    flux_derivative: Float[np.ndarray, " trajectories windows states"],
) -> Float[np.ndarray, " trajectories windows states"]:
    """Return, on the states of every context, the derivative of the cubic of the
    training box that is closest to the true flux derivative."""
    fitted = np.empty_like(flux_derivative)
    for index in np.ndindex(states.shape[:2]):
        coefficients = best_cubic_in_box(
            states[index].astype(np.float64), flux_derivative[index].astype(np.float64)
        )
        fitted[index] = cubic_flux_derivative(coefficients, states[index])
    return fitted


def mismatches(
    seeds: list[EffectiveFlux],
    true_flux: Float[np.ndarray, " trajectories windows states"],
    true_derivative: Float[np.ndarray, " trajectories windows states"],
    box_cubic_derivative: Float[np.ndarray, " trajectories windows states"] | None,
) -> dict[str, Float[np.ndarray, " seeds trajectories windows"]]:
    """Compute the mismatches that the module docstring lists, for every seed."""

    def centered(values):
        return values - values.mean(axis=-1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        derivative_norm = np.sqrt(
            np.trapezoid(true_derivative**2, axis=-1) / (true_derivative.shape[-1] - 1)
        )
    result = {
        "derivative": [
            relative_l2_on_grid(e.derivative - true_derivative, true_derivative)
            for e in seeds
        ],
        "flux": [
            relative_l2_on_grid(
                centered(e.flux) - centered(true_flux), centered(true_flux)
            )
            for e in seeds
        ],
        "spread": [e.derivative_spread / derivative_norm for e in seeds],
        "real_stencil": [e.real_stencil_mismatch for e in seeds],
    }
    if box_cubic_derivative is not None:
        box_cubic = relative_l2_on_grid(
            box_cubic_derivative - true_derivative, true_derivative
        )
        result["box_cubic"] = [box_cubic for _ in seeds]
        result["to_box_cubic"] = [
            relative_l2_on_grid(e.derivative - box_cubic_derivative, true_derivative)
            for e in seeds
        ]
    return {name: np.stack(values) for name, values in result.items()}


def write_results(
    path: Path,
    config: dict[str, Any],
    test_set: LoadedTestSet,
    coefficients: Float[np.ndarray, " trajectories params"],
    coefficient_names: list[str],
    box_cubic_derivative: np.ndarray | None,
    results: dict[str, list[EffectiveFlux]],
    all_mismatches: dict[str, dict[str, np.ndarray]],
    checkpoints: dict[str, list[Path]],
) -> None:
    """Write the effective flux of every context and its mismatches to one file.

    Each model is a group. The fields of `EffectiveFlux` have the seeds as their
    first axis, followed by `(trajectories, windows)`, and the group `mismatches`
    holds one dataset of shape `(seeds, trajectories, windows)` per mismatch. The
    states are the same for every seed and are stored once.
    """
    with h5py.File(path, "w") as f:
        f.attrs["config"] = json.dumps(config)
        f.attrs["model_args"] = test_set.model_args
        f.attrs["num_trajectories_in_file"] = test_set.num_in_file
        f.attrs["coefficient_names"] = coefficient_names
        f.attrs["window_starts"] = config["window_starts"]
        f.create_dataset("trajectory_indices", data=test_set.trajectory_indices)
        f.create_dataset("coefficients", data=coefficients)
        f.create_dataset("states", data=next(iter(results.values()))[0].states)
        if box_cubic_derivative is not None:
            f.create_dataset("box_cubic_derivative", data=box_cubic_derivative)
        for name, seeds in results.items():
            group = f.create_group(f"models/{name}")
            group.attrs["checkpoints"] = [str(p) for p in checkpoints[name]]
            for field in dataclasses.fields(EffectiveFlux):
                if field.name != "states":
                    group.create_dataset(
                        field.name,
                        data=np.stack([getattr(e, field.name) for e in seeds]),
                    )
            for mismatch_name, values in all_mismatches[name].items():
                group.create_dataset(f"mismatches/{mismatch_name}", data=values)


def print_summary(all_mismatches: dict[str, dict[str, np.ndarray]]) -> None:
    """Print the median and the mean of every mismatch over the contexts.

    Both are taken over the contexts with a finite mismatch, per seed, and a cell
    shows the mean and the standard deviation over the seeds. A mismatch is not finite
    on a context whose true flux derivative vanishes on all its states.
    """
    for name, model_mismatches in all_mismatches.items():
        print(f"\n{name}, relative L2 mismatch over the states of a context")
        print(f"{'':14s}{'median':>22s}{'mean':>22s}{'finite fraction':>18s}")
        for mismatch_name, values in model_mismatches.items():
            flat = values.reshape(values.shape[0], -1)
            medians = np.asarray([np.median(v[np.isfinite(v)]) for v in flat])
            means = np.asarray([np.mean(v[np.isfinite(v)]) for v in flat])
            print(
                f"{mismatch_name:14s}"
                f"{medians.mean():>12.3e} ± {medians.std():.1e}"
                f"{means.mean():>12.3e} ± {means.std():.1e}"
                f"{np.isfinite(flat).mean():>18.4f}"
            )
        if "box_cubic" in model_mismatches:
            model = model_mismatches["derivative"]
            box_cubic = model_mismatches["box_cubic"]
            is_finite = np.isfinite(model) & np.isfinite(box_cubic)
            closer = ((model < box_cubic) & is_finite).sum(axis=(1, 2)) / is_finite.sum(
                axis=(1, 2)
            )
            print(
                "Fraction of contexts on which the effective flux is closer to f' "
                f"than the best cubic of the training box: {closer.mean():.3f} ± "
                f"{closer.std():.3f}"
            )


def parse_config() -> dict[str, Any]:
    """Merge the command line, the preset and the defaults, in that order."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", choices=sorted(PRESETS))
    parser.add_argument("--name", help="Stem of the output file. Default: the preset.")
    parser.add_argument("--data-root", help="Directory that --data-path is under.")
    parser.add_argument(
        "--data-path", help="The test file. An absolute path ignores --data-root."
    )
    parser.add_argument(
        "--time-stride", type=int, help="Keep every N-th stored frame of the file."
    )
    parser.add_argument("--checkpoint-root")
    parser.add_argument(
        "--checkpoint",
        action="append",
        metavar="NAME=PATH",
        help="A checkpoint directory under --checkpoint-root. Repeat it for every "
        "seed.",
    )
    parser.add_argument(
        "--window-starts",
        type=int,
        nargs="+",
        help="The frames at which the contexts begin.",
    )
    parser.add_argument("--context-length", type=int)
    parser.add_argument(
        "--num-states", type=int, help="The number of constant states per context."
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gpu", type=int, help="Index of the GPU to run on.")
    given = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}

    preset = PRESETS[given["preset"]] if "preset" in given else {}
    config = DEFAULTS | preset | given
    config.setdefault("name", config.get("preset"))
    required = ("name", "data_path", "checkpoint", "window_starts")
    missing = [key for key in required if config.get(key) is None]
    if missing:
        parser.error(f"Without a preset, these arguments are required: {missing}")
    return config


def main() -> None:
    config = parse_config()
    if config["gpu"] is not None:
        jax.config.update("jax_default_device", jax.devices("gpu")[config["gpu"]])

    data_path = Path(config["data_root"]) / config["data_path"]
    test_set = load_legacy_hdf5(data_path, config["time_stride"], config["limit"])
    trajectories = test_set.trajectories
    coefficients, coefficient_names = load_coefficients(data_path, test_set)
    if tuple(coefficient_names) not in FLUX_FAMILIES:
        raise ValueError(
            f"No flux is known for the coefficients {coefficient_names}. The known "
            f"coefficient names are {sorted(FLUX_FAMILIES)}."
        )
    flux_fn, flux_derivative_fn = FLUX_FAMILIES[tuple(coefficient_names)]
    window_starts = tuple(config["window_starts"])
    print(
        f"{config['name']}: {trajectories.shape[0]} trajectories of shape "
        f"{trajectories.shape[1:]}, {len(window_starts)} window starts, coefficients "
        f"{coefficient_names}"
    )

    checkpoints = resolve_checkpoints(
        config["checkpoint"], Path(config["checkpoint_root"])
    )
    results: dict[str, list[EffectiveFlux]] = {}
    for name, paths in checkpoints.items():
        results[name] = []
        for path in paths:
            print(f"{name}: {path}")
            model = eqx.nn.inference_mode(load_model(path), True)
            results[name].append(
                evaluate_effective_flux(
                    model,
                    trajectories,
                    coefficients.astype(np.float32),
                    window_starts,
                    context_length=config["context_length"],
                    flux_derivative_fn=flux_derivative_fn,
                    num_states=config["num_states"],
                    chunk_size=config["chunk_size"],
                    batch_size=config["batch_size"],
                )
            )

    # The states depend on the context only, so every checkpoint has the same.
    states = next(iter(results.values()))[0].states
    true_flux = true_flux_on_states(flux_fn, coefficients, states)
    true_derivative = true_flux_on_states(flux_derivative_fn, coefficients, states)
    box_cubic_derivative = None
    if tuple(coefficient_names) != CUBIC_COEFFICIENT_NAMES:
        print("Fitting the best cubic of the training box on every context.")
        box_cubic_derivative = fit_box_cubics(states, true_derivative)
    all_mismatches = {
        name: mismatches(seeds, true_flux, true_derivative, box_cubic_derivative)
        for name, seeds in results.items()
    }

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if config["limit"] is None else f"_limit{config['limit']}"
    output_path = output_dir / f"{config['name']}{suffix}.hdf5"
    write_results(
        output_path,
        config,
        test_set,
        coefficients,
        coefficient_names,
        box_cubic_derivative,
        results,
        all_mismatches,
        checkpoints,
    )
    print(f"Wrote {output_path}")
    print_summary(all_mismatches)


if __name__ == "__main__":
    main()
