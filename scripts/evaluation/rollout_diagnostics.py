"""Rollout diagnostics of trained checkpoints on one test set.

This script computes the diagnostics D1 to D6 of the ICLR 2027 rebuttal plan (action
2b) and writes them to one HDF5 file. Each test trajectory gives one rollout: the first
`--context-length` frames are the context, and the model predicts every remaining frame.

A preset holds the test set and the checkpoints of the submission:

    python scripts/evaluation/rollout_diagnostics.py --preset euler_2d

Every value of a preset is also a command-line argument, and an argument overrides the
preset. A test set without a preset, such as 2D cubic, is described by arguments alone:

    python scripts/evaluation/rollout_diagnostics.py \
        --name cubicflux_2d --loader legacy_hdf5 \
        --data-path /path/to/2d_cubic_test_100_100.hdf5 \
        --checkpoint-root /path/to/checkpoints/cubicflux_2d \
        --checkpoint "DPOT=DPOT/OneStepLoss/seed=0/*" \
        --checkpoint "DPOT=DPOT/OneStepLoss/seed=10/*" \
        --checkpoint "HyperFluxFNOLocal=HyperFluxFNOLocal/OneStepLoss/seed=0/*" \
        --batch-size 200

Checkpoints that share a name are the seeds of one model, in the order given.
`--limit N` restricts a run to the first N trajectories and writes to a separate file,
for a trial.
"""

import argparse
import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import equinox as eqx
import h5py
import jax
import numpy as np
import xarray as xr
from context_flux_no.evaluation import (
    euler_minima,
    evaluate_references,
    evaluate_rollouts,
    RolloutEvaluation,
    shallow_water_minima,
    summarize_conservation_drift,
    summarize_rollouts,
)
from context_flux_no.metrics import RolloutReferences
from context_flux_no.training.io import load_model
from jaxtyping import Bool, Float


# The thresholds of the error area fraction (D4). The plan reports 0.1 and uses the
# other two as a sensitivity check.
AREA_THRESHOLDS = (0.05, 0.1, 0.2)
# The rollout steps of the error spectrum (D6): rollout time 0.1, 0.2 and 0.3 at
# dt = 0.005. The spectrum is computed for test sets with two spatial dimensions.
SPECTRUM_STEPS = (20, 40, 60)
# The number of rollout steps that enter the printed summary.
SUMMARY_HORIZON = 80
SUMMARY_STEPS = (20, 40, 80)

MINIMA_FNS: dict[str, tuple[Callable | None, tuple[str, ...]]] = {
    "none": (None, ()),
    "shallow_water": (shallow_water_minima, ("h",)),
    "euler": (euler_minima, ("rho", "p")),
}


def _seeds(prefix: str, paths: dict[str, list[str]]) -> list[str]:
    """Write a model -> checkpoint paths table as `NAME=PATH` entries."""
    return [f"{name}={prefix}/{path}" for name, ps in paths.items() for path in ps]


_CUBIC_1D_CHECKPOINTS = _seeds(
    "cubicflux_1d",
    {
        "DPOT": [
            "DPOT/OneStepLoss/seed=0/26-04-10-14_49_05",
            "DPOT/OneStepLoss/seed=10/26-04-16-18_37_37",
            "DPOT/OneStepLoss/seed=20/26-04-17-07_57_49",
        ],
        "HyperFluxFNOLocal": [
            "HyperFluxFNOLocal/OneStepLoss/seed=0/26-04-16-00_31_47",
            "HyperFluxFNOLocal/OneStepLoss/seed=10/26-04-17-07_57_49",
            "HyperFluxFNOLocal/OneStepLoss/seed=20/26-04-17-10_50_08",
        ],
        "DISCO": [
            "DISCO/OneStepLoss/seed=0/26-04-10-21_26_22",
            "DISCO/OneStepLoss/seed=10/26-04-16-18_37_50",
            "DISCO/OneStepLoss/seed=20/26-04-16-18_37_50",
        ],
    },
)
# The HFluxNO checkpoints of 2D Euler are of class HyperNeuralOperator. They keep the
# name HyperFluxFNOLocal, under which euler_2d_long_rollout.hdf5 stores them.
_EULER_2D_CHECKPOINTS = _seeds(
    "euler_2d",
    {
        "DPOT": [
            "DPOT/OneStepLoss/seed=0/26-09-22-02_45_52",
            "DPOT/OneStepLoss/seed=10/26-09-22-09_38_04",
            "DPOT/OneStepLoss/seed=20/26-09-22-09_38_15",
        ],
        "HyperFluxFNOLocal": [
            "HyperNeuralOperator/OneStepLoss/seed=0/26-09-22-13_04_48",
            "HyperNeuralOperator/OneStepLoss/seed=10/26-09-23-17_31_47",
            "HyperNeuralOperator/OneStepLoss/seed=20/26-09-24-23_34_31",
        ],
        "DISCO": [
            "DISCO/OneStepLoss/seed=0/26-09-22-02_32_26",
            "DISCO/OneStepLoss/seed=10/26-09-24-01_14_36",
            "DISCO/OneStepLoss/seed=20/26-09-24-01_22_00",
        ],
    },
)


def _one_run_per_seed(family: str) -> list[str]:
    """The checkpoints of a family whose seed directories each hold one run."""
    return [
        f"{model}={family}/{model}/OneStepLoss/seed={seed}/*"
        for model in ("DPOT", "HyperFluxFNOLocal", "DISCO")
        for seed in (0, 10, 20)
    ]


# The test sets and checkpoints of the submission, as the evaluation notebooks load
# them. Data paths are relative to --data-root and checkpoints to --checkpoint-root.
PRESETS: dict[str, dict[str, Any]] = {
    "cubic_1d": {
        "loader": "legacy_hdf5",
        "data_path": "datasets/cubic_no_source/data/test/"
        "cubic_no_source_large_test_seed=10.hdf5",
        "time_stride": 10,
        "checkpoint": _CUBIC_1D_CHECKPOINTS,
        "reference_file": "analysis/cubic_1d_long_rollout.hdf5",
        "chunk_size": 10000,
        "batch_size": 2000,
    },
    "cubic_1d_shock": {
        "loader": "legacy_hdf5",
        "data_path": "datasets/cubic_no_source/data/test/"
        "cubic_no_source_large_test_step_seed=10.hdf5",
        "time_stride": 10,
        "checkpoint": _CUBIC_1D_CHECKPOINTS,
        "chunk_size": 10000,
        "batch_size": 2000,
    },
    "shallow_water_1d": {
        "loader": "legacy_hdf5",
        "data_path": "datasets/shallow_water_1d/data/test/"
        "shallow_water_1d_test_grf_seed=10.hdf5",
        "checkpoint": _one_run_per_seed("shallow_water_1d"),
        "positivity": "shallow_water",
        "reference_file": "analysis/shallow_water_1d_long_rollout.hdf5",
        "chunk_size": 10000,
        "batch_size": 1000,
    },
    # The shock set that generate_shallow_water_step_1d.py writes with the coefficients
    # and the m(0) of the training set. The shock sets of the submission had beta in
    # [8, 12], ten times the training range, and are not evaluated here.
    "shallow_water_1d_shock": {
        "loader": "legacy_hdf5",
        "data_path": "datasets/shallow_water_1d/data/test/"
        "shallow_water_1d_test_step_v2_grf_seed=10.hdf5",
        "checkpoint": _one_run_per_seed("shallow_water_1d"),
        "positivity": "shallow_water",
        "chunk_size": 10000,
        "batch_size": 1000,
    },
    "burgers_1d": {
        "loader": "legacy_hdf5",
        "data_path": "datasets/burgers_1d/data/test/burgers_1d_test_grf_seed=10.hdf5",
        "checkpoint": _one_run_per_seed("burgers_1d"),
        "reference_file": "analysis/burgers_1d_long_rollout.hdf5",
        "chunk_size": 10000,
        "batch_size": 1000,
    },
    "burgers_1d_shock": {
        "loader": "legacy_hdf5",
        "data_path": "datasets/burgers_1d/data/test/burgers_1d_test_step_seed=10.hdf5",
        "checkpoint": _one_run_per_seed("burgers_1d"),
        "chunk_size": 10000,
        "batch_size": 1000,
    },
    "euler_2d": {
        "loader": "well_zarr",
        "data_path": "datasets",
        "well_dataset_name": "euler_2d",
        "well_exclude_filters": ["step"],
        "checkpoint": _EULER_2D_CHECKPOINTS,
        "positivity": "euler",
        "coefficient_name": "gamma",
        "reference_file": "analysis/euler_2d_long_rollout.hdf5",
        "chunk_size": 200,
        "batch_size": 50,
    },
    "euler_2d_shock": {
        "loader": "well_zarr",
        "data_path": "datasets",
        "well_dataset_name": "euler_2d",
        "well_exclude_filters": ["grf"],
        "checkpoint": _EULER_2D_CHECKPOINTS,
        "positivity": "euler",
        "coefficient_name": "gamma",
        "chunk_size": 200,
        "batch_size": 50,
    },
}

# The values that hold when neither the command line nor the preset sets them.
DEFAULTS: dict[str, Any] = {
    "data_root": "./data",
    "checkpoint_root": "./checkpoints",
    "output_dir": "./data/analysis/rollout_diagnostics",
    "time_stride": 1,
    "well_dataset_name": None,
    "well_exclude_filters": [],
    "model_args": None,
    "positivity": "none",
    "coefficient_name": None,
    "context_length": 20,
    "reference_file": None,
    "exclude_constant_references": False,
    "limit": None,
    "chunk_size": 1000,
    "batch_size": 100,
    "gpu": None,
}


@dataclasses.dataclass(frozen=True)
class LoadedTestSet:
    """A test set in the layout that `evaluate_rollouts` takes.

    **Attributes:**

    - `trajectories`: The trajectories whose reference solution is finite.
    - `trajectory_indices`: The position of each kept trajectory in the file.
    - `num_in_file`: The number of trajectories in the file.
    - `model_args`: `(dt, dx)` or `(dt, dx, dy)`, read from the grid of the file.
    - `coefficients`: One coefficient per kept trajectory, or `None`.
    """

    trajectories: Float[np.ndarray, " trajectories frames channels *spatial_dims"]
    trajectory_indices: np.ndarray
    num_in_file: int
    model_args: tuple[float, ...]
    coefficients: Float[np.ndarray, " trajectories"] | None


def _keep_valid(
    values: Float[np.ndarray, " trajectories frames channels *spatial_dims"],
    failed: Bool[np.ndarray, " trajectories"],
    model_args: tuple[float, ...],
    coefficients: Float[np.ndarray, " trajectories"] | None,
    limit: int | None,
) -> LoadedTestSet:
    """Drop the trajectories without a finite reference and apply `limit`."""
    num_in_file = values.shape[0]
    if limit is not None:
        values, failed = values[:limit], failed[:limit]
        coefficients = None if coefficients is None else coefficients[:limit]
    is_finite = np.isfinite(values).all(axis=tuple(range(1, values.ndim)))
    keep = is_finite & ~failed
    if not keep.all():
        print(
            f"Left out {int((~keep).sum())} of {len(keep)} trajectories: "
            f"{int(failed.sum())} marked as failed and "
            f"{int((~is_finite & ~failed).sum())} with a non-finite reference."
        )
        values = values[keep]
        coefficients = None if coefficients is None else coefficients[keep]
    return LoadedTestSet(
        trajectories=values,
        trajectory_indices=np.flatnonzero(keep),
        num_in_file=num_in_file,
        model_args=model_args,
        coefficients=coefficients,
    )


def load_legacy_hdf5(path: Path, time_stride: int, limit: int | None) -> LoadedTestSet:
    """Load a test set in the xarray/HDF5 layout of the 1D families and 2D cubic.

    The variable `values` ends in the axes `(t, channel, x)` or `(t, channel, x, y)`.
    All axes before them, such as `(pde, ic)`, are merged into one trajectory axis.
    `time_stride` keeps every `time_stride`-th frame, which brings the files that store
    all solver frames to the time step of the models.
    """
    dataset = xr.open_dataset(path, engine="h5netcdf", chunks={})
    dataset = dataset.isel(t=slice(None, None, time_stride))
    spatial_coordinates = [name for name in ("x", "y") if name in dataset.coords]
    spacing = lambda name: float(dataset[name][1] - dataset[name][0])  # noqa: E731
    model_args = (spacing("t"), *[spacing(name) for name in spatial_coordinates])

    values = dataset["values"].values.astype(np.float32)
    trajectory_shape = values.shape[-(2 + len(spatial_coordinates)) :]
    values = values.reshape(-1, *trajectory_shape)
    if "failed" in dataset:
        failed = dataset["failed"].values.astype(bool).reshape(-1)
    else:
        failed = np.zeros(values.shape[0], dtype=bool)
    return _keep_valid(values, failed, model_args, None, limit)


def load_well_zarr(
    base_path: Path,
    dataset_name: str,
    exclude_filters: list[str],
    coefficient_name: str | None,
    limit: int | None,
) -> LoadedTestSet:
    """Load the test split of a zarr dataset in the Well layout, as used for 2D Euler.

    `coefficient_name` names the per-trajectory scalar, such as `gamma`, that the
    positivity check needs. The data source does not return scalars, so they are read
    from the `scalars` group of each file.
    """
    # Imported here because the module imports grain and the Well, which the legacy
    # test sets do not need.
    from context_flux_no.data.sources import ZarrWellDatasetSource

    source = ZarrWellDatasetSource(
        base_path,
        dataset_name,
        "test",
        exclude_filters=exclude_filters,
        preload_into_ram=True,
    )
    # Each cached file is (trajectory, t, *spatial, channel).
    arrays = [np.moveaxis(cache, -1, 2) for cache in source.preload_cache]
    values = arrays[0] if len(arrays) == 1 else np.concatenate(arrays)

    coefficients = None
    if coefficient_name is not None:
        per_file = []
        for datapath in source.datapaths:
            with source.open_datapath(datapath) as file:
                per_file.append(np.asarray(file["scalars"][coefficient_name][...]))
        coefficients = np.concatenate(per_file).astype(values.dtype)
        if len(coefficients) != values.shape[0]:
            raise ValueError(
                f"Read {len(coefficients)} values of {coefficient_name} for "
                f"{values.shape[0]} trajectories."
            )

    model_args = (
        float(source.metadata_common["temporal_resolution"]),
        *[float(dx) for dx in source.metadata_common["spatial_resolution"]],
    )
    failed = np.zeros(values.shape[0], dtype=bool)
    return _keep_valid(values, failed, model_args, coefficients, limit)


def resolve_checkpoints(
    entries: list[str], checkpoint_root: Path
) -> dict[str, list[Path]]:
    """Turn `NAME=PATH` entries into the checkpoint directories of each model.

    A path may contain `*`, which must then match exactly one directory.
    """
    checkpoints: dict[str, list[Path]] = {}
    for entry in entries:
        name, separator, pattern = entry.partition("=")
        if not separator or not name or not pattern:
            raise ValueError(f"--checkpoint expects NAME=PATH, got {entry!r}.")
        matches = sorted(checkpoint_root.glob(pattern))
        if len(matches) != 1:
            raise ValueError(
                f"{pattern!r} matches {len(matches)} paths under {checkpoint_root}, "
                "expected exactly one."
            )
        checkpoints.setdefault(name, []).append(matches[0])
    return checkpoints


def write_results(
    path: Path,
    config: dict[str, Any],
    test_set: LoadedTestSet,
    model_args: tuple[float, ...],
    spectrum_steps: tuple[int, ...],
    minima_names: tuple[str, ...],
    references: RolloutReferences,
    results: dict[str, list[RolloutEvaluation]],
    checkpoints: dict[str, list[Path]],
) -> None:
    """Write the reference quantities and the results of every model to one file.

    Each model is a group whose datasets have the seeds as their first axis.
    """
    with h5py.File(path, "w") as f:
        f.attrs["config"] = json.dumps(config)
        f.attrs["model_args"] = model_args
        f.attrs["area_thresholds"] = AREA_THRESHOLDS
        f.attrs["spectrum_steps"] = spectrum_steps
        f.attrs["minima_names"] = list(minima_names)
        f.attrs["num_trajectories_in_file"] = test_set.num_in_file
        f.create_dataset("trajectory_indices", data=test_set.trajectory_indices)

        group = f.create_group("references")
        for field in dataclasses.fields(references):
            group.create_dataset(field.name, data=getattr(references, field.name))

        for name, evaluations in results.items():
            group = f.create_group(name)
            group.attrs["checkpoints"] = [str(p) for p in checkpoints[name]]
            for field in dataclasses.fields(evaluations[0].diagnostics):
                stacked = np.stack(
                    [getattr(e.diagnostics, field.name) for e in evaluations]
                )
                group.create_dataset(field.name, data=stacked)
            for field_name in ("minima", "error_spectrum", "num_finite_spectra"):
                if getattr(evaluations[0], field_name) is not None:
                    stacked = np.stack([getattr(e, field_name) for e in evaluations])
                    group.create_dataset(field_name, data=stacked)


def compare_with_stored(
    reference_file: Path,
    test_set: LoadedTestSet,
    results: dict[str, list[RolloutEvaluation]],
) -> None:
    """Print the largest difference between the new l2 error and a stored one.

    The stored file is a `{family}_long_rollout.hdf5` of the submission, with one
    `l2` dataset of shape `(seed, trajectory, step)` per model. The comparison covers
    the steps that both hold.
    """
    with h5py.File(reference_file, "r") as f:
        for name, evaluations in results.items():
            if name not in f:
                print(f"  {name}: no group of this name in {reference_file.name}")
                continue
            stored = f[name]["l2"][()]
            new = np.stack([e.diagnostics.l2 for e in evaluations])
            if stored.shape[0] != new.shape[0] or stored.shape[1] < new.shape[1]:
                print(
                    f"  {name}: stored shape {stored.shape} cannot be matched to "
                    f"the new shape {new.shape}"
                )
                continue
            if stored.shape[1] == test_set.num_in_file:
                stored = stored[:, test_set.trajectory_indices]
            else:
                # The stored file already leaves out the trajectories without a
                # reference, so the kept ones are matched by position.
                stored = stored[:, : new.shape[1]]
            num_steps = min(stored.shape[2], new.shape[2])
            stored, new = stored[..., :num_steps], new[..., :num_steps]
            both_finite = np.isfinite(stored) & np.isfinite(new)
            with np.errstate(invalid="ignore"):
                difference = np.abs(new - stored)[both_finite]
            relative = difference / np.abs(stored[both_finite])
            print(
                f"  {name}: {num_steps} steps, largest difference "
                f"{difference.max():.2e} absolute and {relative.max():.2e} relative, "
                f"{int((~both_finite).sum())} entries not finite in both"
            )


def print_summary(
    results: dict[str, list[RolloutEvaluation]],
    references: RolloutReferences,
    exclude_constant_references: bool,
) -> None:
    """Print the summary of D2 to D5 for every model, pooled over its seeds.

    `exclude_constant_references` leaves the trajectories with a constant reference
    out of the summary, as described in `summarize_rollouts`. The result file holds
    every trajectory either way.
    """
    num_steps = references.l2_identity.shape[-1]
    horizon = min(SUMMARY_HORIZON, num_steps)
    report_steps = tuple(step for step in SUMMARY_STEPS if step <= horizon)
    if horizon < SUMMARY_HORIZON:
        print(f"The rollouts have {num_steps} steps, so the summary uses {horizon}.")
    for name, evaluations in results.items():
        stack = lambda field: np.stack(  # noqa: E731
            [getattr(e.diagnostics, field) for e in evaluations]
        )
        minima = None
        if evaluations[0].minima is not None:
            minima = np.stack([e.minima for e in evaluations])
        summary = summarize_rollouts(
            stack("l2"),
            stack("nonfinite"),
            stack("area_fraction"),
            stack("total_variation"),
            references,
            exclude_constant_references=exclude_constant_references,
            minima=minima,
            horizon=horizon,
            report_steps=report_steps,
        )
        quartiles = " / ".join(f"{q:.0f}" for q in summary["valid_horizon_quartiles"])
        print(
            f"{name} ({summary['num_rollouts']} rollouts; "
            f"{summary['num_constant_references']} trajectories with a constant "
            f"reference, {'left out' if exclude_constant_references else 'included'})"
        )
        print(
            f"  valid horizon, 25 / 50 / 75%: {quartiles}"
            f"   never crossed: {summary['fraction_never_crossed']:.1%}"
            f"   crossed at step 1: {summary['fraction_crossed_at_step_1']:.1%}"
        )
        rows = {
            "non-finite": summary["fraction_nonfinite"],
            "diverged": summary["fraction_diverged"],
            "non-positive": summary["fraction_nonpositive"],
            **{
                f"median area fraction, tau = {tau}": medians
                for tau, medians in zip(
                    AREA_THRESHOLDS, summary["median_area_fraction"]
                )
            },
            "median total-variation ratio": summary["median_total_variation_ratio"],
        }
        print(f"  {'by step':34s}" + "".join(f"{step:>10d}" for step in report_steps))
        for label, row in rows.items():
            if row is not None:
                cells = "".join(f"{row[step]:>10.3g}" for step in report_steps)
                print(f"  {label:34s}{cells}")

    print_conservation_drift(results, references, horizon)


def print_conservation_drift(
    results: dict[str, list[RolloutEvaluation]],
    references: RolloutReferences,
    horizon: int,
) -> None:
    """Print the conservation drift of the reference and of every model.

    Each entry is the median over the rollouts, pooled over the seeds of a model, with
    the 25th and 75th percentiles in brackets. The three columns are the numbers of
    `summarize_conservation_drift`.
    """
    kinds = ("mean", "final", "worst")
    drifts = {"reference": references.conservation_drift} | {
        name: np.stack([e.diagnostics.conservation_drift for e in evaluations])
        for name, evaluations in results.items()
    }
    summaries = {
        name: summarize_conservation_drift(drift, horizon=horizon)
        for name, drift in drifts.items()
    }
    print(f"Conservation drift over {horizon} steps, median [25%, 75%] over rollouts")
    for channel in range(references.conservation_drift.shape[-1]):
        print(f"  channel {channel:<12d}" + "".join(f"{kind:>34s}" for kind in kinds))
        for name, summary in summaries.items():
            cells = ""
            for kind in kinds:
                low, median, high = summary[kind][:, channel]
                cells += f"{median:>12.2e} [{low:.2e}, {high:.2e}]"
            finite = summary["fraction_finite"][channel]
            note = "" if finite == 1.0 else f"   finite: {finite:.1%} of rollouts"
            print(f"  {name:20s}{cells}{note}")


def parse_config() -> dict[str, Any]:
    """Merge the command line, the preset and the defaults, in that order."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", choices=sorted(PRESETS))
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
    parser.add_argument("--checkpoint-root")
    parser.add_argument(
        "--checkpoint",
        action="append",
        metavar="NAME=PATH",
        help="A checkpoint directory under --checkpoint-root. Repeat it for every "
        "seed and model.",
    )
    parser.add_argument(
        "--model-args",
        type=float,
        nargs="+",
        help="The args of model.rollout. Default: (dt, dx[, dy]) from the file.",
    )
    parser.add_argument("--positivity", choices=sorted(MINIMA_FNS))
    parser.add_argument("--coefficient-name")
    parser.add_argument("--context-length", type=int)
    parser.add_argument(
        "--reference-file",
        help="A stored {family}_long_rollout.hdf5 to compare l2 with.",
    )
    parser.add_argument(
        "--exclude-constant-references",
        action=argparse.BooleanOptionalAction,
        help="Leave the trajectories with a constant reference out of the printed "
        "summary of D2 to D5. The result file holds every trajectory either way.",
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
    missing = [
        key
        for key in ("name", "loader", "data_path", "checkpoint")
        if config.get(key) is None
    ]
    if missing:
        parser.error(f"Without a preset, these arguments are required: {missing}")
    if config["loader"] == "well_zarr" and config["well_dataset_name"] is None:
        parser.error("--loader well_zarr requires --well-dataset-name")
    if config["positivity"] == "euler" and config["coefficient_name"] is None:
        parser.error("--positivity euler requires --coefficient-name")
    return config


def main() -> None:
    config = parse_config()
    if config["gpu"] is not None:
        jax.config.update("jax_default_device", jax.devices("gpu")[config["gpu"]])

    data_path = Path(config["data_root"]) / config["data_path"]
    if config["loader"] == "legacy_hdf5":
        test_set = load_legacy_hdf5(data_path, config["time_stride"], config["limit"])
    else:
        test_set = load_well_zarr(
            data_path,
            config["well_dataset_name"],
            config["well_exclude_filters"],
            config["coefficient_name"],
            config["limit"],
        )
    trajectories = test_set.trajectories
    model_args = tuple(config["model_args"] or test_set.model_args)
    num_spatial_dims = trajectories.ndim - 3
    num_steps = trajectories.shape[1] - config["context_length"]
    spectrum_steps = SPECTRUM_STEPS if num_spatial_dims == 2 else ()
    minima_fn, minima_names = MINIMA_FNS[config["positivity"]]
    print(
        f"{config['name']}: {trajectories.shape[0]} trajectories of shape "
        f"{trajectories.shape[1:]}, {num_steps} rollout steps, model args {model_args}"
    )

    checkpoints = resolve_checkpoints(
        config["checkpoint"], Path(config["checkpoint_root"])
    )
    references = evaluate_references(
        trajectories,
        context_length=config["context_length"],
        chunk_size=config["chunk_size"],
    )
    results: dict[str, list[RolloutEvaluation]] = {}
    for name, paths in checkpoints.items():
        results[name] = []
        for path in paths:
            print(f"{name}: {path}")
            model = eqx.nn.inference_mode(load_model(path), True)
            results[name].append(
                evaluate_rollouts(
                    model,
                    trajectories,
                    model_args,
                    context_length=config["context_length"],
                    area_thresholds=AREA_THRESHOLDS,
                    spectrum_steps=spectrum_steps,
                    minima_fn=minima_fn,
                    coefficients=test_set.coefficients,
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
        model_args,
        spectrum_steps,
        minima_names,
        references,
        results,
        checkpoints,
    )
    print(f"Wrote {output_path}")

    if config["reference_file"] is not None:
        reference_file = Path(config["data_root"]) / config["reference_file"]
        if reference_file.exists():
            print(f"Comparison of l2 with {reference_file}:")
            compare_with_stored(reference_file, test_set, results)
        else:
            print(f"{reference_file} does not exist, so l2 is not compared.")

    print_summary(results, references, config["exclude_constant_references"])


if __name__ == "__main__":
    main()
