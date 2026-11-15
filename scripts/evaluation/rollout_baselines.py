"""Errors of the model-free baselines on the rollouts of one test set.

This script computes the reference curves of Fig. 3(a) and writes them to one HDF5 file.
Each test trajectory gives one rollout, as in `rollout_diagnostics.py`: the first
`--context-length` frames are the context, and the remaining frames are the reference.
The file holds two relative l2 errors per trajectory and rollout step:

- `l2_identity`: every frame is predicted by the last frame of the context, which is
  the rollout of a model that returns its last input frame.
- `l2_spatial_mean`: every frame is predicted by the cell mean of each channel of the
  last frame of the context.

The script loads no model and no checkpoint, so it needs the test set alone. A preset of
`rollout_diagnostics.py` names a test set of the submission:

    python scripts/evaluation/rollout_baselines.py --preset euler_2d

A test set without a preset, such as 2D cubic, is described by arguments:

    python scripts/evaluation/rollout_baselines.py \
        --name cubicflux_2d --loader legacy_hdf5 \
        --data-path /path/to/2d_cubic_test_100_100.hdf5

`--stored-file` names the `{family}_long_rollout.hdf5` of the submission, and the script
then reports whether its trajectories and steps match those of the new file.
`notebooks/evaluation/long_term_rollouts.ipynb` reads the files.

The loaders are those of `rollout_diagnostics.py` in this directory, which Python finds
because it puts the directory of a script on the path.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import jax
import numpy as np
from context_flux_no.evaluation import evaluate_references
from jaxtyping import Float
from rollout_diagnostics import (
    load_legacy_hdf5,
    load_well_zarr,
    LoadedTestSet,
    PRESETS as ROLLOUT_PRESETS,
)


# The values that hold when neither the command line nor the preset sets them.
DEFAULTS: dict[str, Any] = {
    "data_root": "./data",
    "output_dir": "./data/analysis/rollout_baselines",
    "time_stride": 1,
    "well_dataset_name": None,
    "well_exclude_filters": [],
    "context_length": 20,
    "stored_file": None,
    "chunk_size": 1000,
    "gpu": None,
}
# The entries of a rollout preset that describe the test set. The others describe the
# checkpoints and the diagnostics, which this script does not use.
PRESET_KEYS = (
    "loader",
    "data_path",
    "time_stride",
    "well_dataset_name",
    "well_exclude_filters",
    "chunk_size",
)


def write_baselines(
    path: Path,
    config: dict[str, Any],
    test_set: LoadedTestSet,
    errors: dict[str, Float[np.ndarray, " trajectories steps"]],
) -> None:
    """Write the baseline errors and the description of the test set to one file."""
    with h5py.File(path, "w") as f:
        f.attrs["config"] = json.dumps(config)
        f.attrs["model_args"] = test_set.model_args
        f.attrs["context_length"] = config["context_length"]
        f.attrs["num_trajectories_in_file"] = test_set.num_in_file
        f.create_dataset("trajectory_indices", data=test_set.trajectory_indices)
        for name, values in errors.items():
            f.create_dataset(name, data=values)


def compare_with_stored(
    stored_file: Path, num_trajectories: int, num_steps: int
) -> None:
    """Print whether a stored `{family}_long_rollout.hdf5` matches the new file.

    The stored file has one `l2` dataset of shape `(seed, trajectory, step)` per model.
    The numbers of trajectories must agree for the curves of Fig. 3(a) to describe the
    same rollouts. The stored file may hold fewer steps, and the figure then uses the
    steps that both files hold.
    """
    with h5py.File(stored_file, "r") as f:
        for name, group in f.items():
            if not isinstance(group, h5py.Group) or "l2" not in group:
                continue
            _, stored_trajectories, stored_steps = group["l2"].shape
            if stored_trajectories != num_trajectories:
                verdict = "MISMATCH in the number of trajectories"
            elif stored_steps > num_steps:
                verdict = "MISMATCH: the stored file holds more steps"
            else:
                verdict = "match"
            print(
                f"  {name}: stored {stored_trajectories} trajectories and "
                f"{stored_steps} steps, new {num_trajectories} and {num_steps}: "
                f"{verdict}"
            )


def parse_config() -> dict[str, Any]:
    """Merge the command line, the preset and the defaults, in that order."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", choices=sorted(ROLLOUT_PRESETS))
    parser.add_argument("--name", help="Stem of the output file. Default: the preset.")
    parser.add_argument("--loader", choices=("legacy_hdf5", "well_zarr"))
    parser.add_argument("--data-root", help="Directory that --data-path is under.")
    parser.add_argument(
        "--data-path",
        help="legacy_hdf5: the test file. well_zarr: the base directory of the "
        "datasets. An absolute path ignores --data-root.",
    )
    parser.add_argument("--time-stride", type=int)
    parser.add_argument("--well-dataset-name")
    parser.add_argument("--well-exclude-filters", nargs="*")
    parser.add_argument("--context-length", type=int)
    parser.add_argument(
        "--stored-file",
        help="A stored {family}_long_rollout.hdf5 to compare the shapes with. An "
        "absolute path ignores --data-root.",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--gpu", type=int, help="Index of the GPU to run on.")
    given = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}

    preset = {}
    if "preset" in given:
        rollout_preset = ROLLOUT_PRESETS[given["preset"]]
        preset = {k: rollout_preset[k] for k in PRESET_KEYS if k in rollout_preset}
    config = DEFAULTS | preset | given
    config.setdefault("name", config.get("preset"))
    missing = [
        key for key in ("name", "loader", "data_path") if config.get(key) is None
    ]
    if missing:
        parser.error(f"Without a preset, these arguments are required: {missing}")
    if config["loader"] == "well_zarr" and config["well_dataset_name"] is None:
        parser.error("--loader well_zarr requires --well-dataset-name")
    return config


def main() -> None:
    config = parse_config()
    if config["gpu"] is not None:
        jax.config.update("jax_default_device", jax.devices("gpu")[config["gpu"]])

    data_path = Path(config["data_root"]) / config["data_path"]
    if config["loader"] == "legacy_hdf5":
        test_set = load_legacy_hdf5(data_path, config["time_stride"], None)
    else:
        test_set = load_well_zarr(
            data_path,
            config["well_dataset_name"],
            config["well_exclude_filters"],
            None,
            None,
        )
    trajectories = test_set.trajectories
    num_trajectories, num_frames = trajectories.shape[:2]
    num_steps = num_frames - config["context_length"]
    print(
        f"{config['name']}: {num_trajectories} trajectories of shape "
        f"{trajectories.shape[1:]}, {num_steps} rollout steps after a context of "
        f"{config['context_length']} frames, (dt, dx, ...) = {test_set.model_args}"
    )

    references = evaluate_references(
        trajectories,
        context_length=config["context_length"],
        chunk_size=config["chunk_size"],
    )
    errors = {
        "l2_identity": references.l2_identity,
        "l2_spatial_mean": references.l2_spatial_mean,
    }
    for name, values in errors.items():
        medians = np.nanmedian(values, axis=0)
        print(
            f"  {name}: median {medians[0]:.3g} at step 1 and {medians[-1]:.3g} at "
            f"step {num_steps}"
        )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config['name']}.hdf5"
    write_baselines(output_path, config, test_set, errors)
    print(f"Wrote {output_path}")

    if config["stored_file"] is not None:
        stored_file = Path(config["data_root"]) / config["stored_file"]
        print(f"Comparison of the shapes with {stored_file}:")
        compare_with_stored(stored_file, num_trajectories, num_steps)


if __name__ == "__main__":
    main()
