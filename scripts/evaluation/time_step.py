"""Single-step errors of HFluxNO at time steps other than the training step.

This script tests how the generated flux behaves at another time step (ICLR 2027
rebuttal plan, action 7c). The context is sampled at the training step and encoded
once. The generated flux networks then advance the last context frame by one step of
several sizes, and each result is compared with the solver frame at that time. The
context is never sampled at another step: the encoder does not receive the time step,
so a context at twice the step looks like a flux that is twice as large.

    python scripts/evaluation/time_step.py --preset cubic_1d

Every value of the preset is also a command-line argument, and an argument overrides
the preset. `--limit N` restricts a run to the first N trajectories and writes to a
separate file, for a trial.

The preset reads the in-distribution set at half the training step, so two stored
frames make one model step, and it takes steps over 1, 2 and 4 stored frames: half the
training step, the training step and twice the training step.

The script writes the errors of every segment to one HDF5 file. It prints the error
relative to the identity baseline for every step size, in bins of the CFL number of
the step. The CFL number is the largest |f'(u)| over the interval of states of the
last context frame, times the step size over dx. The bins are shared by the step
sizes, so one row compares the step sizes at equal CFL numbers.

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
import numpy as np
from context_flux_no.evaluation import (
    equal_count_bins,
    evaluate_time_steps,
    max_cubic_wave_speed,
    mean_by_bin,
    TimeStepErrors,
)
from context_flux_no.training.io import load_model
from context_length import CUBIC_COEFFICIENT_NAMES, load_coefficients, state_intervals
from jaxtyping import Float
from rollout_diagnostics import load_legacy_hdf5, LoadedTestSet, resolve_checkpoints


# The number of bins of the CFL number in the printed table.
NUM_CFL_BINS = 12

_HFLUXNO = "cubicflux_1d/HyperFluxFNOLocal/OneStepLoss"
# Data paths are relative to --data-root and checkpoints to --checkpoint-root. The
# file stores all solver frames at a tenth of the training step, so a stride of 5
# keeps frames at half the training step. The window starts are those of the
# unseen-flux tests (action 1b), in model steps.
PRESETS: dict[str, dict[str, Any]] = {
    "cubic_1d": {
        "data_path": "datasets/cubic_no_source/data/test/"
        "cubic_no_source_large_test_seed=10.hdf5",
        "time_stride": 5,
        "frames_per_step": 2,
        "step_frames": [1, 2, 4],
        "checkpoint": [
            f"HyperFluxFNOLocal={_HFLUXNO}/seed=0/26-04-16-00_31_47",
            f"HyperFluxFNOLocal={_HFLUXNO}/seed=10/26-04-17-07_57_49",
            f"HyperFluxFNOLocal={_HFLUXNO}/seed=20/26-04-17-10_50_08",
        ],
        "window_starts": list(range(2, 60, 6)),
    },
}

DEFAULTS: dict[str, Any] = {
    "data_root": "./data",
    "checkpoint_root": "./checkpoints",
    "output_dir": "./data/analysis/time_step",
    "context_length": 20,
    "limit": None,
    "chunk_size": 5000,
    "batch_size": 200,
    "gpu": None,
}


def _stack_seeds(seeds: list[TimeStepErrors]) -> dict[str, np.ndarray]:
    return {
        field.name: np.stack([getattr(errors, field.name) for errors in seeds])
        for field in dataclasses.fields(TimeStepErrors)
    }


def write_results(
    path: Path,
    config: dict[str, Any],
    test_set: LoadedTestSet,
    coefficients: Float[np.ndarray, " trajectories 3"],
    cfl: Float[np.ndarray, " trajectories windows steps"],
    results: dict[str, dict[str, np.ndarray]],
    checkpoints: dict[str, list[Path]],
) -> None:
    """Write the per-segment errors of every model and the CFL number of every step.

    Each model is a group whose datasets have the shape
    `(seeds, trajectories, windows, step sizes)`.
    """
    with h5py.File(path, "w") as f:
        f.attrs["config"] = json.dumps(config)
        f.attrs["model_args"] = test_set.model_args
        f.attrs["num_trajectories_in_file"] = test_set.num_in_file
        f.attrs["window_starts"] = config["window_starts"]
        f.attrs["frames_per_step"] = config["frames_per_step"]
        f.attrs["step_frames"] = config["step_frames"]
        f.create_dataset("trajectory_indices", data=test_set.trajectory_indices)
        f.create_dataset("coefficients", data=coefficients)
        f.create_dataset("cfl", data=cfl)
        for name, fields in results.items():
            group = f.create_group(f"models/{name}")
            group.attrs["checkpoints"] = [str(p) for p in checkpoints[name]]
            for field_name, values in fields.items():
                group.create_dataset(field_name, data=values)


def print_summary(
    cfl: Float[np.ndarray, " trajectories windows steps"],
    results: dict[str, dict[str, np.ndarray]],
    step_frames: list[int],
    frames_per_step: int,
) -> None:
    """Print the l2 error relative to the identity baseline for every step size, over
    all segments and in bins of the CFL number.

    The bins hold equally many entries of all step sizes together. A cell is the mean
    error of the model in the bin divided by the mean error of the identity baseline
    in the bin, averaged over the seeds, followed by the number of segments of that
    step size in the bin. A non-finite error makes the cell of its bin non-finite.
    """
    labels = [f"{m / frames_per_step:g} dt" for m in step_frames]
    bin_indices, edges = equal_count_bins(cfl.reshape(-1), NUM_CFL_BINS)
    bin_indices = bin_indices.reshape(cfl.shape)
    for name, fields in results.items():
        num_seeds = fields["l2"].shape[0]
        print(f"\n{name}, l2 error of one step relative to the identity baseline")
        header = "".join(f"{label:>20s}" for label in labels)
        print(f"{'CFL number':>18s}{header}")
        cells = ""
        for i in range(len(step_frames)):
            model = fields["l2"][..., i].reshape(num_seeds, -1)
            identity = fields["l2_identity"][..., i].reshape(num_seeds, -1)
            is_finite = np.isfinite(model) & np.isfinite(identity)
            relative = np.where(is_finite, model, 0.0).sum(axis=1) / np.where(
                is_finite, identity, 0.0
            ).sum(axis=1)
            cells += f"{relative.mean():>11.3f} ± {relative.std():.3f}"
        print(f"{'all segments':>18s}{cells}")
        per_step = []
        for i in range(len(step_frames)):
            indices = bin_indices[..., i].reshape(-1)
            model, counts = mean_by_bin(
                fields["l2"][..., i].reshape(num_seeds, -1), indices, NUM_CFL_BINS
            )
            identity, _ = mean_by_bin(
                fields["l2_identity"][..., i].reshape(num_seeds, -1),
                indices,
                NUM_CFL_BINS,
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                per_step.append(((model / identity).mean(axis=0), counts))
        for b in range(NUM_CFL_BINS):
            cells = "".join(
                f"{relative[b]:>11.3f} ({counts[b]:>6d})"
                for relative, counts in per_step
            )
            print(f"{edges[b]:>8.2f} - {edges[b + 1]:<7.2f}{cells}")


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
    parser.add_argument(
        "--frames-per-step",
        type=int,
        help="The number of kept frames per time step of the model.",
    )
    parser.add_argument(
        "--step-frames",
        type=int,
        nargs="+",
        help="The step sizes, in kept frames.",
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
        help="The model steps at which the contexts begin.",
    )
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gpu", type=int, help="Index of the GPU to run on.")
    given = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}

    preset = PRESETS[given["preset"]] if "preset" in given else {}
    config = DEFAULTS | preset | given
    config.setdefault("name", config.get("preset"))
    required = (
        "name",
        "data_path",
        "time_stride",
        "frames_per_step",
        "step_frames",
        "checkpoint",
        "window_starts",
    )
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
    if tuple(coefficient_names) != CUBIC_COEFFICIENT_NAMES:
        raise ValueError(
            f"The CFL number is defined for the cubic flux with the coefficients "
            f"{CUBIC_COEFFICIENT_NAMES}, got {coefficient_names}."
        )
    window_starts = tuple(config["window_starts"])
    context_length = config["context_length"]
    frames_per_step = config["frames_per_step"]
    step_frames = list(config["step_frames"])
    dt_stored, dx = test_set.model_args
    print(
        f"{config['name']}: {trajectories.shape[0]} trajectories of shape "
        f"{trajectories.shape[1:]}, {len(window_starts)} window starts, model step "
        f"{frames_per_step * dt_stored:g}, step sizes "
        f"{[m * dt_stored for m in step_frames]}, dx {dx:g}"
    )

    last_context_frames = tuple(
        (start + context_length - 1) * frames_per_step for start in window_starts
    )
    state_min, state_max = state_intervals(trajectories, last_context_frames)
    wave_speed = max_cubic_wave_speed(coefficients[:, None, :], state_min, state_max)
    cfl = wave_speed[..., None] * np.asarray(step_frames) * dt_stored / dx

    checkpoints = resolve_checkpoints(
        config["checkpoint"], Path(config["checkpoint_root"])
    )
    results: dict[str, dict[str, np.ndarray]] = {}
    for name, paths in checkpoints.items():
        seeds = []
        for path in paths:
            print(f"{name}: {path}")
            model = eqx.nn.inference_mode(load_model(path), True)
            seeds.append(
                evaluate_time_steps(
                    model,
                    trajectories,
                    window_starts,
                    test_set.model_args,
                    context_length=context_length,
                    frames_per_step=frames_per_step,
                    step_frames=tuple(step_frames),
                    chunk_size=config["chunk_size"],
                    batch_size=config["batch_size"],
                )
            )
        results[name] = _stack_seeds(seeds)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if config["limit"] is None else f"_limit{config['limit']}"
    output_path = output_dir / f"{config['name']}{suffix}.hdf5"
    write_results(
        output_path, config, test_set, coefficients, cfl, results, checkpoints
    )
    print(f"Wrote {output_path}")
    print_summary(cfl, results, step_frames, frames_per_step)


if __name__ == "__main__":
    main()
