"""Errors of trained checkpoints on every segment of a shallow-water test set.

This script produces the numbers behind actions 5a and 5b of the ICLR 2027 rebuttal
plan, the diagnosis of the shallow-water loss under shock-dominated initial conditions.
It repeats the evaluation of `notebooks/evaluation/evaluate_shallow.ipynb`, in which a
segment is 20 context frames followed by 20 predicted frames, and keeps the errors of
each segment where the notebook averages them. It also stores what the diagnosis groups
the segments by:

- the errors of the identity baseline and of the spatial-mean baseline,
- the number of jumps and the mean of h in the first frame of each trajectory,
- the largest jump of h in the last context frame of each segment,
- the coefficients and the first frame of each trajectory, which pair two test sets.

A preset holds one test set and the checkpoints of the submission:

    python scripts/evaluation/segment_errors.py --preset shallow_water_1d_shock

Every value of a preset is also a command-line argument, and an argument overrides the
preset. `--limit N` restricts a run to the first N trajectories and writes to a
separate file, for a trial. `notebooks/evaluation/shallow_water_shock_diagnosis.ipynb`
reads the files.

The loader and the checkpoint resolution are those of `rollout_diagnostics.py` in this
directory, which Python finds because it puts the directory of a script on the path.
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
import xarray as xr
from context_flux_no.evaluation import (
    evaluate_segment_baselines,
    evaluate_segments,
    SegmentBaselines,
)
from context_flux_no.metrics import count_jumps, largest_jump, SegmentErrors
from context_flux_no.training.io import load_model
from jaxtyping import Float
from rollout_diagnostics import (
    load_legacy_hdf5,
    LoadedTestSet,
    PRESETS as ROLLOUT_PRESETS,
    resolve_checkpoints,
)


# The window starts of the new test sets of the rebuttal plan (decision of 2026-11-10
# for action 1b, reused for the low set of action 5b).
SPARSE_WINDOW_STARTS = tuple(range(2, 60, 6))

# The test sets of the diagnosis. Data paths are relative to --data-root and
# checkpoints to --checkpoint-root. The first two paths are those that
# `evaluate_shallow.ipynb` loads (cells 2 and 12).
PRESETS: dict[str, dict[str, Any]] = {
    "shallow_water_1d": {
        "data_path": "datasets/shallow_water_1d/data/test/"
        "shallow_water_1d_test_grf_seed=10.hdf5",
    },
    "shallow_water_1d_shock": {
        "data_path": "shallow_water_1d_test_step_grf_seed=10.hdf5",
    },
    "shallow_water_1d_shock_low": {
        "data_path": "datasets/shallow_water_1d/data/test/"
        "shallow_water_1d_test_step_low_grf_seed=10.hdf5",
        "window_starts": list(SPARSE_WINDOW_STARTS),
    },
}

# The values that hold when neither the command line nor the preset sets them.
DEFAULTS: dict[str, Any] = {
    "data_root": "./data",
    "checkpoint_root": "./checkpoints",
    "checkpoint": ROLLOUT_PRESETS["shallow_water_1d"]["checkpoint"],
    "output_dir": "./data/analysis/segment_errors",
    "window_starts": None,
    "context_length": 20,
    "rollout_steps": 20,
    "limit": None,
    "chunk_size": 1000,
    "batch_size": 16,
    "gpu": None,
}


def load_coefficients(
    path: Path, test_set: LoadedTestSet
) -> tuple[Float[np.ndarray, " trajectories params"], list[str]]:
    """Read the coefficients of the trajectories that `load_legacy_hdf5` kept.

    The shallow-water files store one row of coefficients per trajectory, in the
    variable `coeffs` with the dimensions `(ic, param)`.
    """
    dataset = xr.open_dataset(path, engine="h5netcdf")
    coeffs = dataset["coeffs"]
    if coeffs.dims != ("ic", "param"):
        raise ValueError(
            f"Expected coeffs with dimensions ('ic', 'param'), got {coeffs.dims}."
        )
    names = [str(name) for name in dataset["param"].values]
    return coeffs.values[test_set.trajectory_indices], names


def front_strengths(
    trajectories: Float[np.ndarray, " trajectories frames channels x"],
    window_starts: tuple[int, ...],
    context_length: int,
) -> Float[np.ndarray, " trajectories windows"]:
    """Return the largest jump of the first channel in the last context frame of every
    segment."""
    last_context_frames = np.asarray(window_starts) + context_length - 1
    return np.asarray(
        jax.vmap(jax.vmap(largest_jump))(
            jnp.asarray(trajectories[:, last_context_frames, 0])
        )
    )


def _write_segment_errors(group: h5py.Group, errors: SegmentErrors) -> None:
    for field in dataclasses.fields(errors):
        group.create_dataset(field.name, data=getattr(errors, field.name))


def write_results(
    path: Path,
    config: dict[str, Any],
    test_set: LoadedTestSet,
    window_starts: tuple[int, ...],
    coefficients: Float[np.ndarray, " trajectories params"],
    coefficient_names: list[str],
    baselines: SegmentBaselines,
    results: dict[str, list[SegmentErrors]],
    checkpoints: dict[str, list[Path]],
) -> None:
    """Write the grouping quantities, the baselines and the errors of every model.

    The datasets of a model have the axes `(seed, trajectory, window)`, and those of a
    baseline the axes `(trajectory, window)`.
    """
    trajectories = test_set.trajectories
    with h5py.File(path, "w") as f:
        f.attrs["config"] = json.dumps(config)
        f.attrs["model_args"] = test_set.model_args
        f.attrs["num_trajectories_in_file"] = test_set.num_in_file
        f.attrs["coefficient_names"] = coefficient_names
        f.create_dataset("trajectory_indices", data=test_set.trajectory_indices)
        f.create_dataset("window_starts", data=np.asarray(window_starts))
        f.create_dataset("coefficients", data=coefficients)
        f.create_dataset("initial_state", data=trajectories[:, 0])
        f.create_dataset(
            "jump_count",
            data=np.asarray(jax.vmap(count_jumps)(jnp.asarray(trajectories[:, 0, 0]))),
        )
        f.create_dataset("mean_depth", data=trajectories[:, 0, 0].mean(axis=-1))
        f.create_dataset(
            "front_strength",
            data=front_strengths(trajectories, window_starts, config["context_length"]),
        )

        group = f.create_group("baselines")
        _write_segment_errors(group.create_group("identity"), baselines.identity)
        _write_segment_errors(
            group.create_group("spatial_mean"), baselines.spatial_mean
        )

        group = f.create_group("models")
        for name, errors in results.items():
            model_group = group.create_group(name)
            model_group.attrs["checkpoints"] = [str(p) for p in checkpoints[name]]
            stacked = jax.tree.map(lambda *xs: np.stack(xs), *errors)
            _write_segment_errors(model_group, stacked)


def print_summary(
    baselines: SegmentBaselines, results: dict[str, list[SegmentErrors]]
) -> None:
    """Print the mean over segments of every error, as mean and standard deviation
    over the seeds of a model. These are the numbers of Tables 12 and 13."""
    fields = [field.name for field in dataclasses.fields(SegmentErrors)]
    print(f"{'':20s}" + "".join(f"{name:>24s}" for name in fields))
    for label, errors in (
        ("identity baseline", baselines.identity),
        ("spatial-mean baseline", baselines.spatial_mean),
    ):
        cells = "".join(f"{getattr(errors, name).mean():>24.3e}" for name in fields)
        print(f"{label:20s}{cells}")
    for name, seeds in results.items():
        cells = ""
        for field in fields:
            per_seed = np.asarray([getattr(e, field).mean() for e in seeds])
            cells += f"{per_seed.mean():>13.3e} ± {per_seed.std():.1e}"
        print(f"{name:20s}{cells}")


def parse_config() -> dict[str, Any]:
    """Merge the command line, the preset and the defaults, in that order."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", choices=sorted(PRESETS))
    parser.add_argument("--name", help="Stem of the output file. Default: the preset.")
    parser.add_argument("--data-root", help="Directory that --data-path is under.")
    parser.add_argument(
        "--data-path", help="The test file. An absolute path ignores --data-root."
    )
    parser.add_argument("--checkpoint-root")
    parser.add_argument(
        "--checkpoint",
        action="append",
        metavar="NAME=PATH",
        help="A checkpoint directory under --checkpoint-root. Repeat it for every "
        "seed and model. Default: the shallow-water checkpoints of the submission.",
    )
    parser.add_argument(
        "--window-starts",
        type=int,
        nargs="+",
        help="The frames at which segments begin. Default: every possible start.",
    )
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gpu", type=int, help="Index of the GPU to run on.")
    given = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}

    preset = PRESETS[given["preset"]] if "preset" in given else {}
    config = DEFAULTS | preset | given
    config.setdefault("name", config.get("preset"))
    missing = [key for key in ("name", "data_path") if config.get(key) is None]
    if missing:
        parser.error(f"Without a preset, these arguments are required: {missing}")
    return config


def main() -> None:
    config = parse_config()
    if config["gpu"] is not None:
        jax.config.update("jax_default_device", jax.devices("gpu")[config["gpu"]])

    data_path = Path(config["data_root"]) / config["data_path"]
    test_set = load_legacy_hdf5(data_path, 1, config["limit"])
    trajectories = test_set.trajectories
    coefficients, coefficient_names = load_coefficients(data_path, test_set)

    segment_length = config["context_length"] + config["rollout_steps"]
    if config["window_starts"] is None:
        window_starts = tuple(range(trajectories.shape[1] - segment_length + 1))
    else:
        window_starts = tuple(config["window_starts"])
    print(
        f"{config['name']}: {trajectories.shape[0]} trajectories of shape "
        f"{trajectories.shape[1:]}, {len(window_starts)} window starts, "
        f"model args {test_set.model_args}"
    )

    checkpoints = resolve_checkpoints(
        config["checkpoint"], Path(config["checkpoint_root"])
    )
    baselines = evaluate_segment_baselines(
        trajectories,
        window_starts,
        context_length=config["context_length"],
        rollout_steps=config["rollout_steps"],
        chunk_size=config["chunk_size"],
    )
    results: dict[str, list[SegmentErrors]] = {}
    for name, paths in checkpoints.items():
        results[name] = []
        for path in paths:
            print(f"{name}: {path}")
            model = eqx.nn.inference_mode(load_model(path), True)
            results[name].append(
                evaluate_segments(
                    model,
                    trajectories,
                    window_starts,
                    test_set.model_args,
                    context_length=config["context_length"],
                    rollout_steps=config["rollout_steps"],
                    chunk_size=config["chunk_size"],
                    batch_size=config["batch_size"],
                )
            )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if config["limit"] is None else f"_limit{config['limit']}"
    output_path = output_dir / f"{config['name']}{suffix}.hdf5"
    write_results(
        output_path,
        config,
        test_set,
        window_starts,
        coefficients,
        coefficient_names,
        baselines,
        results,
        checkpoints,
    )
    print(f"Wrote {output_path}")
    print_summary(baselines, results)


if __name__ == "__main__":
    main()
