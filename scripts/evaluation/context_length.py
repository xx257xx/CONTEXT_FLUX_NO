"""Segment errors of HFluxNO checkpoints at several context lengths.

This script computes the context-length matrix and the CFL analysis of the ICLR 2027
rebuttal plan (actions 7a to 7c) and writes them to one HDF5 file per test set. Every
checkpoint is evaluated at every length of `--test-context-lengths`. The checkpoints of
one name are the seeds of one model, such as the model trained at a context of 5 frames.

A segment is given by its last context frame, not by its first. A context of k frames
that ends at frame e starts at frame e - k + 1, and the model predicts the frames after
e. All context lengths therefore predict the same frames, and the identity baseline is
the same for all of them. `ablate_context_length.ipynb` fixed the first frame instead,
so a context of 5 frames predicted frames 35 frames earlier than a context of 40.

The matrix on the in-distribution set and on one sine set:

    python scripts/evaluation/context_length.py --preset cubic_1d
    python scripts/evaluation/context_length.py --preset sine_rescaled_2-3_grf

The CFL analysis evaluates the submitted checkpoints of all three models at a context
of 20 frames, at the window starts of the unseen-flux tests:

    python scripts/evaluation/context_length.py --preset cubic_1d_cfl

Every value of a preset is also a command-line argument, and an argument overrides the
preset. `--limit N` restricts a run to the first N trajectories and writes to a
separate file, for a trial.

For a cubic test set the script also stores the CFL number of every segment and prints
the single-step error in bins of it. The CFL number is the largest |f'(u)| over the
interval of states of the last context frame, times dt / dx.

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
import numpy as np
import xarray as xr
from context_flux_no.evaluation import (
    equal_count_bins,
    evaluate_segment_baselines,
    evaluate_segments,
    max_cubic_wave_speed,
    mean_by_bin,
    SegmentBaselines,
)
from context_flux_no.metrics import SegmentErrors
from context_flux_no.training.io import load_model
from jaxtyping import Float
from omegaconf import OmegaConf
from orbax.checkpoint import v1 as ocp
from rollout_diagnostics import (
    load_legacy_hdf5,
    LoadedTestSet,
    PRESETS as ROLLOUT_PRESETS,
    resolve_checkpoints,
)


# The last context frames of the matrix. With at most 40 context frames and 20
# predicted frames, every segment lies within frames 0 to 77, inside the 80 frames
# that the checkpoints were trained on.
ALIGNED_WINDOW_ENDS = (39, 45, 51, 57)
# The last context frames of the window starts 2, 8, ..., 56 of the unseen-flux tests
# (action 1b) at a context of 20 frames.
UNSEEN_FLUX_WINDOW_ENDS = tuple(start + 19 for start in range(2, 60, 6))
# The number of bins of the CFL number in the printed table.
NUM_CFL_BINS = 10
# The entries of a stored model configuration in which the checkpoints of one model
# class may differ: the training context length and the seed of the initialization.
ALLOWED_CONFIG_DIFFERENCES = frozenset({"in_timesteps", "key.seed"})
# The names of the cubic coefficients, in the order of `max_cubic_wave_speed`.
CUBIC_COEFFICIENT_NAMES = ("a", "b", "c")

_HFLUXNO = "cubicflux_1d/HyperFluxFNOLocal/OneStepLoss"
# The HFluxNO checkpoints by training context length, with seeds 0, 10 and 20 for
# every length. The lengths 5, 10 and 40 were trained for the NeurIPS rebuttal
# (`notebooks/misc/ablate_context_length.ipynb`). The length 20 is the model of the
# submission.
_CONTEXT_LENGTH_CHECKPOINTS = [
    f"trained_at_5={_HFLUXNO}/seed=0/26-07-24-17_02_51",
    f"trained_at_5={_HFLUXNO}/seed=10/26-07-25-03_23_49",
    f"trained_at_5={_HFLUXNO}/seed=20/26-07-25-14_07_23",
    f"trained_at_10={_HFLUXNO}/seed=0/26-07-24-23_54_13",
    f"trained_at_10={_HFLUXNO}/seed=10/26-07-25-07_25_05",
    f"trained_at_10={_HFLUXNO}/seed=20/26-07-25-15_45_14",
    f"trained_at_20={_HFLUXNO}/seed=0/26-04-16-00_31_47",
    f"trained_at_20={_HFLUXNO}/seed=10/26-04-17-07_57_49",
    f"trained_at_20={_HFLUXNO}/seed=20/26-04-17-10_50_08",
    f"trained_at_40={_HFLUXNO}/seed=0/26-07-25-02_56_19",
    f"trained_at_40={_HFLUXNO}/seed=10/26-07-25-10_56_49",
    f"trained_at_40={_HFLUXNO}/seed=20/26-07-25-16_04_56",
]

_TEST_DIR = "datasets/cubic_no_source/data/test"
_MATRIX = {
    "checkpoint": _CONTEXT_LENGTH_CHECKPOINTS,
    "test_context_lengths": [5, 10, 20, 40],
    "window_ends": list(ALIGNED_WINDOW_ENDS),
}

# Data paths are relative to --data-root and checkpoints to --checkpoint-root. The
# files of the submission store all solver frames, and a stride of 10 brings them to
# the time step of the models. The sine sets of action 1b store every 10th frame.
PRESETS: dict[str, dict[str, Any]] = {
    "cubic_1d": {
        "data_path": f"{_TEST_DIR}/cubic_no_source_large_test_seed=10.hdf5",
        "time_stride": 10,
        **_MATRIX,
    },
    **{
        f"sine_rescaled_{b_bin}_grf": {
            "data_path": f"{_TEST_DIR}/sine_rescaled_{b_bin}_test_grf_seed=13.hdf5",
            "time_stride": 1,
            **_MATRIX,
        }
        for b_bin in ("0-1", "1-2", "2-3", "3-4")
    },
    "cubic_1d_cfl": {
        "data_path": f"{_TEST_DIR}/cubic_no_source_large_test_seed=10.hdf5",
        "time_stride": 10,
        "checkpoint": ROLLOUT_PRESETS["cubic_1d"]["checkpoint"],
        "test_context_lengths": [20],
        "window_ends": list(UNSEEN_FLUX_WINDOW_ENDS),
    },
}

DEFAULTS: dict[str, Any] = {
    "data_root": "./data",
    "checkpoint_root": "./checkpoints",
    "output_dir": "./data/analysis/context_length",
    "time_stride": 1,
    "rollout_steps": 20,
    "limit": None,
    "chunk_size": 10000,
    "batch_size": 500,
    "gpu": None,
}


def load_coefficients(
    path: Path, test_set: LoadedTestSet
) -> tuple[Float[np.ndarray, " trajectories params"], list[str]]:
    """Read the coefficients of the trajectories that `load_legacy_hdf5` kept.

    The 1D scalar files store one row of coefficients per coefficient draw, in the
    variable `coeffs` with the dimensions `(pde, param)`, and `values` begins with the
    dimensions `(pde, ic)`. Every initial condition of a draw gets the row of its draw.
    """
    dataset = xr.open_dataset(path, engine="h5netcdf", chunks={})
    if "coeffs" not in dataset or dataset["coeffs"].dims != ("pde", "param"):
        raise ValueError(
            f"Expected a variable coeffs with dimensions ('pde', 'param') in {path}. "
            f"The file has the variables "
            f"{ {name: var.dims for name, var in dataset.data_vars.items()} }."
        )
    if dataset["values"].dims[:2] != ("pde", "ic"):
        raise ValueError(
            f"Expected values to begin with the dimensions ('pde', 'ic'), got "
            f"{dataset['values'].dims}."
        )
    per_trajectory = np.repeat(dataset["coeffs"].values, dataset.sizes["ic"], axis=0)
    names = [str(name) for name in dataset["param"].values]
    return per_trajectory[test_set.trajectory_indices], names


def read_model_config(checkpoint: Path) -> dict[str, Any]:
    """Return the model configuration that a checkpoint stores, with one entry per
    leaf and the path of the leaf, such as `encoder_kwargs.depth`, as its key."""
    with ocp.training.Checkpointer(checkpoint.resolve()) as manager:
        config = OmegaConf.create(manager.root_metadata().custom_metadata)
    flat: dict[str, Any] = {}

    def visit(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                visit(value, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                visit(value, f"{prefix}[{index}]")
        else:
            flat[prefix] = node

    visit(OmegaConf.to_container(config, resolve=True), "")
    return flat


def check_model_configs(configs: dict[Path, dict[str, Any]]) -> None:
    """Compare the model configuration of every checkpoint with the first checkpoint
    of the same model class.

    A difference in one of `ALLOWED_CONFIG_DIFFERENCES` is printed. Any other
    difference raises, because the checkpoints would then differ in more than their
    training context length and their seed.
    """
    first_of_class: dict[Any, Path] = {}
    unexpected = []
    for path, config in configs.items():
        reference_path = first_of_class.setdefault(config.get("_target_"), path)
        reference = configs[reference_path]
        for key in sorted(reference.keys() | config.keys()):
            if reference.get(key) == config.get(key):
                continue
            line = (
                f"  {path}: {key} = {config.get(key, '<absent>')!r}, "
                f"against {reference.get(key, '<absent>')!r} in {reference_path}"
            )
            if key in ALLOWED_CONFIG_DIFFERENCES:
                print(line)
            else:
                unexpected.append(line)
    if unexpected:
        raise ValueError(
            "The stored model configurations differ in more than "
            f"{sorted(ALLOWED_CONFIG_DIFFERENCES)}:\n" + "\n".join(unexpected)
        )


def window_starts_for(
    window_ends: tuple[int, ...], context_length: int
) -> tuple[int, ...]:
    """Return the first frames of the contexts of `context_length` frames that end at
    `window_ends`."""
    starts = tuple(end - context_length + 1 for end in window_ends)
    if min(starts) < 0:
        raise ValueError(
            f"A context of {context_length} frames that ends at frame "
            f"{min(window_ends)} would start before frame 0."
        )
    return starts


def state_intervals(
    trajectories: Float[np.ndarray, " trajectories frames channels x"],
    window_ends: tuple[int, ...],
) -> tuple[
    Float[np.ndarray, " trajectories windows"],
    Float[np.ndarray, " trajectories windows"],
]:
    """Return the smallest and the largest state of the last context frame of every
    segment."""
    last_frames = trajectories[:, list(window_ends)]
    return last_frames.min(axis=(2, 3)), last_frames.max(axis=(2, 3))


def _stack_fields(per_item: list[SegmentErrors]) -> dict[str, np.ndarray]:
    """Stack the fields of several `SegmentErrors` along a new first axis."""
    return {
        field.name: np.stack([getattr(errors, field.name) for errors in per_item])
        for field in dataclasses.fields(SegmentErrors)
    }


def write_results(
    path: Path,
    config: dict[str, Any],
    test_set: LoadedTestSet,
    coefficients: Float[np.ndarray, " trajectories params"],
    coefficient_names: list[str],
    state_min: Float[np.ndarray, " trajectories windows"],
    state_max: Float[np.ndarray, " trajectories windows"],
    cfl: Float[np.ndarray, " trajectories windows"] | None,
    baselines: SegmentBaselines,
    results: dict[str, dict[str, np.ndarray]],
    checkpoints: dict[str, list[Path]],
) -> None:
    """Write the per-segment errors of every model and the quantities that group them.

    Each model is a group whose datasets have the shape
    `(seeds, test context lengths, trajectories, windows)`. The baselines have the
    shape `(trajectories, windows)`, because they do not depend on the context length.
    """
    with h5py.File(path, "w") as f:
        f.attrs["config"] = json.dumps(config)
        f.attrs["model_args"] = test_set.model_args
        f.attrs["num_trajectories_in_file"] = test_set.num_in_file
        f.attrs["coefficient_names"] = coefficient_names
        f.attrs["window_ends"] = config["window_ends"]
        f.attrs["test_context_lengths"] = config["test_context_lengths"]
        f.create_dataset("trajectory_indices", data=test_set.trajectory_indices)
        f.create_dataset("coefficients", data=coefficients)
        f.create_dataset("state_min", data=state_min)
        f.create_dataset("state_max", data=state_max)
        if cfl is not None:
            f.create_dataset("cfl", data=cfl)

        for baseline_name in ("identity", "spatial_mean"):
            group = f.create_group(f"baselines/{baseline_name}")
            errors = getattr(baselines, baseline_name)
            for field in dataclasses.fields(SegmentErrors):
                group.create_dataset(field.name, data=getattr(errors, field.name))

        for name, fields in results.items():
            group = f.create_group(f"models/{name}")
            group.attrs["checkpoints"] = [str(p) for p in checkpoints[name]]
            for field_name, values in fields.items():
                group.create_dataset(field_name, data=values)


def _mean_over_finite(
    values: Float[np.ndarray, " *batch trajectories windows"],
    is_finite: np.ndarray,
) -> Float[np.ndarray, " *batch"]:
    """Average over the segments that `is_finite` marks."""
    return np.where(is_finite, values, 0.0).sum(axis=(-2, -1)) / is_finite.sum(
        axis=(-2, -1)
    )


def print_matrix(
    baselines: SegmentBaselines,
    results: dict[str, dict[str, np.ndarray]],
    test_context_lengths: list[int],
) -> None:
    """Print, for the single-step and the rollout l2 error, the mean of every model at
    every test context length, and that mean relative to the identity baseline.

    A segment enters a cell when the error of the model and of the identity baseline
    are finite on it. The mean is taken per seed, and a cell shows the mean and the
    standard deviation over the seeds. The fraction of segments left out is printed
    when it is not zero.
    """
    for field in ("l2_onestep", "l2_rollout"):
        identity = getattr(baselines.identity, field)
        header = "".join(f"{f'k = {k}':>24s}" for k in test_context_lengths)
        print(f"\n{field}, mean over segments (relative to the identity baseline)")
        print(f"{'':16s}{header}")
        for name, fields in results.items():
            values = fields[field]
            is_finite = np.isfinite(values) & np.isfinite(identity)
            per_seed = _mean_over_finite(values, is_finite)
            relative = per_seed / _mean_over_finite(
                np.broadcast_to(identity, values.shape), is_finite
            )
            cells = "".join(
                f"{per_seed[:, i].mean():>10.3e} ± {per_seed[:, i].std():.0e}"
                f" ({relative[:, i].mean():.2f})"
                for i in range(len(test_context_lengths))
            )
            print(f"{name:16s}{cells}")
            left_out = 1 - is_finite.mean(axis=(0, 2, 3))
            if left_out.any():
                print(
                    f"{'':16s}fraction of segments left out: "
                    + ", ".join(f"{fraction:.2e}" for fraction in left_out)
                )


def print_cfl_table(
    cfl: Float[np.ndarray, " trajectories windows"],
    baselines: SegmentBaselines,
    results: dict[str, dict[str, np.ndarray]],
    test_context_lengths: list[int],
) -> None:
    """Print the single-step l2 error relative to the identity baseline in bins of the
    CFL number that hold equally many segments, for every test context length.

    A cell is the mean error of the model in the bin divided by the mean error of the
    identity baseline in the bin, averaged over the seeds. A non-finite error makes the
    cell of its bin non-finite.
    """
    bin_indices, edges = equal_count_bins(cfl.reshape(-1), NUM_CFL_BINS)
    identity, counts = mean_by_bin(
        baselines.identity.l2_onestep.reshape(-1), bin_indices, NUM_CFL_BINS
    )
    for i, context_length in enumerate(test_context_lengths):
        print(
            f"\nl2_onestep relative to the identity baseline by CFL number, "
            f"k = {context_length}"
        )
        names = "".join(f"{name:>20s}" for name in results)
        print(f"{'CFL number':>16s}{'segments':>10s}{'identity':>11s}{names}")
        per_model = []
        for fields in results.values():
            values = fields["l2_onestep"][:, i].reshape(
                fields["l2_onestep"].shape[0], -1
            )
            means, _ = mean_by_bin(values, bin_indices, NUM_CFL_BINS)
            per_model.append((means / identity).mean(axis=0))
        for b in range(NUM_CFL_BINS):
            cells = "".join(f"{relative[b]:>20.3f}" for relative in per_model)
            print(
                f"{edges[b]:>7.2f} - {edges[b + 1]:<6.2f}{counts[b]:>10d}"
                f"{identity[b]:>11.2e}{cells}"
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
        "seed and model.",
    )
    parser.add_argument(
        "--test-context-lengths",
        type=int,
        nargs="+",
        help="The context lengths at which every checkpoint is evaluated.",
    )
    parser.add_argument(
        "--window-ends",
        type=int,
        nargs="+",
        help="The last context frame of every segment.",
    )
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
    required = (
        "name",
        "data_path",
        "checkpoint",
        "test_context_lengths",
        "window_ends",
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
    window_ends = tuple(config["window_ends"])
    test_context_lengths = list(config["test_context_lengths"])
    rollout_steps = config["rollout_steps"]
    print(
        f"{config['name']}: {trajectories.shape[0]} trajectories of shape "
        f"{trajectories.shape[1:]}, last context frames {window_ends}, test context "
        f"lengths {test_context_lengths}, model args {test_set.model_args}"
    )

    checkpoints = resolve_checkpoints(
        config["checkpoint"], Path(config["checkpoint_root"])
    )
    print("Differences between the stored model configurations:")
    check_model_configs(
        {
            path: read_model_config(path)
            for paths in checkpoints.values()
            for path in paths
        }
    )

    # The baselines depend on the last context frame and on the predicted frames only,
    # so one context length gives them for all.
    baselines = evaluate_segment_baselines(
        trajectories,
        window_starts_for(window_ends, test_context_lengths[0]),
        context_length=test_context_lengths[0],
        rollout_steps=rollout_steps,
        chunk_size=config["chunk_size"],
    )
    state_min, state_max = state_intervals(trajectories, window_ends)
    cfl = None
    if tuple(coefficient_names) == CUBIC_COEFFICIENT_NAMES:
        dt, dx = test_set.model_args
        cfl = max_cubic_wave_speed(coefficients[:, None, :], state_min, state_max)
        cfl = cfl * dt / dx

    results: dict[str, dict[str, np.ndarray]] = {}
    for name, paths in checkpoints.items():
        per_seed = []
        for path in paths:
            print(f"{name}: {path}")
            model = eqx.nn.inference_mode(load_model(path), True)
            per_length = [
                evaluate_segments(
                    model,
                    trajectories,
                    window_starts_for(window_ends, context_length),
                    test_set.model_args,
                    context_length=context_length,
                    rollout_steps=rollout_steps,
                    chunk_size=config["chunk_size"],
                    batch_size=config["batch_size"],
                )
                for context_length in test_context_lengths
            ]
            per_seed.append(_stack_fields(per_length))
        results[name] = {
            field: np.stack([seed[field] for seed in per_seed]) for field in per_seed[0]
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
        state_min,
        state_max,
        cfl,
        baselines,
        results,
        checkpoints,
    )
    print(f"Wrote {output_path}")
    print_matrix(baselines, results, test_context_lengths)
    if cfl is not None:
        print_cfl_table(cfl, baselines, results, test_context_lengths)


if __name__ == "__main__":
    main()
