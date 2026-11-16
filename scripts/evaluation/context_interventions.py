"""Errors of HFluxNO when its encoder reads a shuffled, a static or a donor context.

This script answers whether shuffling or replacing the context affects the prediction
(ICLR 2027 rebuttal plan, action 3a). On every segment of a 1D cubic test set the flux
networks are generated from one of these contexts, and the true last context frame is
advanced with them:

- `true`: the context of the segment.
- `shuffled`: the same frames in a random order.
- `static`: copies of the last context frame, which carry no dynamics.
- `donor_same`: the context of another initial condition with the same coefficients.
- `donor_other_0`, ...: the contexts of trajectories with other coefficients.

A donor context covers the same frames as the context of the segment.

    python scripts/evaluation/context_interventions.py --preset cubic_1d

Every value of the preset is also a command-line argument, and an argument overrides
the preset. `--limit N` restricts a run to the first N trajectories and writes to a
separate file, for a trial.

The script writes the errors of every segment to one HDF5 file and prints their means.
`l2_onestep` is the error of one step. `l2_rollout` is the error of a rollout in which
the flux networks are held for all steps, so the `true` row is not the rollout of the
submission, which regenerates the flux networks at every step.

For a donor with other coefficients the script stores the distance between the flux
derivatives of donor and receiver, ||f'_donor - f'_receiver|| / ||f'_receiver||, with
L2 norms over the interval of states of the receiver's last context frame. It prints
the error in bins of this distance.

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
    cubic_wave_speed_distance,
    equal_count_bins,
    evaluate_context_interventions,
    evaluate_segment_baselines,
    INTERVENTIONS,
    mean_by_bin,
    SegmentBaselines,
)
from context_flux_no.metrics import SegmentErrors
from context_flux_no.training.io import load_model
from context_length import CUBIC_COEFFICIENT_NAMES, load_coefficients, state_intervals
from jaxtyping import Float, Int
from rollout_diagnostics import load_legacy_hdf5, LoadedTestSet, resolve_checkpoints


# The number of bins of the flux-derivative distance in the printed table.
NUM_DISTANCE_BINS = 8

_HFLUXNO = "cubicflux_1d/HyperFluxFNOLocal/OneStepLoss"
# Data paths are relative to --data-root and checkpoints to --checkpoint-root. The
# window starts are those of the unseen-flux tests (action 1b).
PRESETS: dict[str, dict[str, Any]] = {
    "cubic_1d": {
        "data_path": "datasets/cubic_no_source/data/test/"
        "cubic_no_source_large_test_seed=10.hdf5",
        "time_stride": 10,
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
    "output_dir": "./data/analysis/context_interventions",
    "time_stride": 1,
    "context_length": 20,
    "rollout_steps": 20,
    "num_other_donors": 4,
    "seed": 0,
    "limit": None,
    "chunk_size": 2000,
    "batch_size": 50,
    "gpu": None,
}


def choose_donors(
    coefficients: Float[np.ndarray, " trajectories params"],
    num_other_donors: int,
    rng: np.random.Generator,
) -> Int[np.ndarray, " trajectories 1+{num_other_donors}"]:
    """Choose the donor trajectories of every trajectory at random.

    The first donor has the same coefficients and is another trajectory. It is -1 when
    no other trajectory has these coefficients. The remaining donors have other
    coefficients and are distinct.
    """
    _, draws = np.unique(coefficients, axis=0, return_inverse=True)
    draws = draws.reshape(-1)
    indices = np.arange(len(draws))
    donors = np.full((len(draws), 1 + num_other_donors), -1, dtype=np.int64)
    for index, draw in enumerate(draws):
        same = indices[(draws == draw) & (indices != index)]
        if same.size:
            donors[index, 0] = rng.choice(same)
        other = indices[draws != draw]
        if other.size < num_other_donors:
            raise ValueError(
                f"Trajectory {index} has {other.size} trajectories with other "
                f"coefficients, fewer than the {num_other_donors} donors asked for."
            )
        donors[index, 1:] = rng.choice(other, size=num_other_donors, replace=False)
    return donors


def intervention_names(num_other_donors: int) -> list[str]:
    """Return the names of the last axis of the errors."""
    others = [f"donor_other_{i}" for i in range(num_other_donors)]
    return [*INTERVENTIONS, "donor_same", *others]


def _stack_seeds(seeds: list[SegmentErrors]) -> dict[str, np.ndarray]:
    return {
        field.name: np.stack([getattr(errors, field.name) for errors in seeds])
        for field in dataclasses.fields(SegmentErrors)
    }


def write_results(
    path: Path,
    config: dict[str, Any],
    test_set: LoadedTestSet,
    coefficients: Float[np.ndarray, " trajectories 3"],
    donors: Int[np.ndarray, " trajectories donors"],
    permutations: Int[np.ndarray, " trajectories windows context_length"],
    distance: Float[np.ndarray, " trajectories windows donors"],
    baselines: SegmentBaselines,
    results: dict[str, dict[str, np.ndarray]],
    checkpoints: dict[str, list[Path]],
) -> None:
    """Write the per-segment errors of every model and the quantities that group them.

    Each model is a group whose datasets have the shape
    `(seeds, trajectories, windows, interventions)`. `donors` holds positions among
    the trajectories of this file, not among those of the test file.
    """
    with h5py.File(path, "w") as f:
        f.attrs["config"] = json.dumps(config)
        f.attrs["model_args"] = test_set.model_args
        f.attrs["num_trajectories_in_file"] = test_set.num_in_file
        f.attrs["window_starts"] = config["window_starts"]
        f.attrs["interventions"] = intervention_names(config["num_other_donors"])
        f.create_dataset("trajectory_indices", data=test_set.trajectory_indices)
        f.create_dataset("coefficients", data=coefficients)
        f.create_dataset("donors", data=donors)
        f.create_dataset("permutations", data=permutations.astype(np.int8))
        f.create_dataset("flux_derivative_distance", data=distance)
        for field in dataclasses.fields(SegmentErrors):
            f.create_dataset(
                f"baselines/identity/{field.name}",
                data=getattr(baselines.identity, field.name),
            )
        for name, fields in results.items():
            group = f.create_group(f"models/{name}")
            group.attrs["checkpoints"] = [str(p) for p in checkpoints[name]]
            for field_name, values in fields.items():
                group.create_dataset(field_name, data=values)


def _mean_over_finite(values: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
    is_finite = np.isfinite(values)
    return np.where(is_finite, values, 0.0).sum(axis=axis) / is_finite.sum(axis=axis)


def print_summary(
    baselines: SegmentBaselines,
    results: dict[str, dict[str, np.ndarray]],
    num_other_donors: int,
) -> None:
    """Print the mean errors of every intervention, with the donors of other
    coefficients pooled into one row.

    The mean is taken over the segments with a finite error, per seed. A cell shows
    the mean and the standard deviation over the seeds and, in parentheses, the mean
    divided by the mean error of the identity baseline over all segments.
    """
    names = intervention_names(num_other_donors)
    rows = {name: [names.index(name)] for name in (*INTERVENTIONS, "donor_same")}
    rows["donor_other"] = list(range(len(INTERVENTIONS) + 1, len(names)))
    fields = ("l2_onestep", "l_inf_onestep", "l2_rollout", "l_inf_rollout")
    for model_name, model_fields in results.items():
        print(f"\n{model_name}, mean over segments (relative to the identity baseline)")
        print(f"{'':14s}" + "".join(f"{field:>31s}" for field in fields))
        for row, columns in rows.items():
            cells = ""
            for field in fields:
                per_seed = _mean_over_finite(
                    model_fields[field][..., columns], axis=(1, 2, 3)
                )
                identity = _mean_over_finite(
                    getattr(baselines.identity, field), axis=(0, 1)
                )
                cells += (
                    f"{per_seed.mean():>13.3e} ± {per_seed.std():.1e}"
                    f" ({per_seed.mean() / identity:>5.2f})"
                )
            print(f"{row:14s}{cells}")
        missing = np.isnan(
            model_fields["l2_onestep"][0, :, 0, names.index("donor_same")]
        )
        if missing.any():
            print(f"{'':14s}{int(missing.sum())} trajectories have no donor_same.")


def print_dose_response(
    distance: Float[np.ndarray, " trajectories windows donors"],
    results: dict[str, dict[str, np.ndarray]],
) -> None:
    """Print the single-step l2 error of the donors with other coefficients in bins of
    the flux-derivative distance that hold equally many entries.

    A cell is the mean error with the donor context in the bin divided by the mean
    error with the true context on the same segments, averaged over the seeds. A
    non-finite error makes the cell of its bin non-finite.
    """
    first_other = len(INTERVENTIONS) + 1
    other_distance = distance[..., 1:]
    bin_indices, edges = equal_count_bins(other_distance.reshape(-1), NUM_DISTANCE_BINS)
    true = INTERVENTIONS.index("true")
    print(
        "\nl2_onestep with a donor of other coefficients, relative to the true "
        "context, by flux-derivative distance"
    )
    print(f"{'distance':>18s}{'entries':>10s}" + "".join(f"{n:>20s}" for n in results))
    per_model, counts = [], None
    for fields in results.values():
        errors = fields["l2_onestep"]
        donor = errors[..., first_other:].reshape(errors.shape[0], -1)
        reference = np.broadcast_to(
            errors[..., true : true + 1], errors[..., first_other:].shape
        ).reshape(errors.shape[0], -1)
        donor_means, counts = mean_by_bin(donor, bin_indices, NUM_DISTANCE_BINS)
        reference_means, _ = mean_by_bin(reference, bin_indices, NUM_DISTANCE_BINS)
        per_model.append((donor_means / reference_means).mean(axis=0))
    for b in range(NUM_DISTANCE_BINS):
        cells = "".join(f"{relative[b]:>20.2f}" for relative in per_model)
        print(f"{edges[b]:>8.3f} - {edges[b + 1]:<7.3f}{counts[b]:>10d}{cells}")


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
        help="The frames at which segments begin.",
    )
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument(
        "--num-other-donors",
        type=int,
        help="The number of donors with other coefficients per trajectory.",
    )
    parser.add_argument(
        "--seed", type=int, help="Seed of the donor choice and of the shuffling."
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
    if tuple(coefficient_names) != CUBIC_COEFFICIENT_NAMES:
        raise ValueError(
            f"The flux-derivative distance is defined for the cubic flux with the "
            f"coefficients {CUBIC_COEFFICIENT_NAMES}, got {coefficient_names}."
        )
    window_starts = tuple(config["window_starts"])
    context_length = config["context_length"]
    segments = dict(
        context_length=context_length, rollout_steps=config["rollout_steps"]
    )
    print(
        f"{config['name']}: {trajectories.shape[0]} trajectories of shape "
        f"{trajectories.shape[1:]}, {len(window_starts)} window starts, "
        f"model args {test_set.model_args}"
    )

    rng = np.random.default_rng(config["seed"])
    donors = choose_donors(coefficients, config["num_other_donors"], rng)
    permutations = rng.permuted(
        np.broadcast_to(
            np.arange(context_length),
            (trajectories.shape[0], len(window_starts), context_length),
        ),
        axis=-1,
    )
    window_ends = tuple(start + context_length - 1 for start in window_starts)
    state_min, state_max = state_intervals(trajectories, window_ends)
    # A missing donor has the index -1, which selects the last trajectory here. Its
    # distance is set to NaN below.
    distance = cubic_wave_speed_distance(
        coefficients[:, None, None, :],
        coefficients[donors][:, None, :, :],
        state_min[:, :, None],
        state_max[:, :, None],
    )
    distance = np.where((donors < 0)[:, None, :], np.nan, distance)

    checkpoints = resolve_checkpoints(
        config["checkpoint"], Path(config["checkpoint_root"])
    )
    baselines = evaluate_segment_baselines(
        trajectories, window_starts, chunk_size=config["chunk_size"], **segments
    )
    results: dict[str, dict[str, np.ndarray]] = {}
    for name, paths in checkpoints.items():
        seeds = []
        for path in paths:
            print(f"{name}: {path}")
            model = eqx.nn.inference_mode(load_model(path), True)
            seeds.append(
                evaluate_context_interventions(
                    model,
                    trajectories,
                    donors,
                    permutations,
                    window_starts,
                    test_set.model_args,
                    chunk_size=config["chunk_size"],
                    batch_size=config["batch_size"],
                    **segments,
                )
            )
        results[name] = _stack_seeds(seeds)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if config["limit"] is None else f"_limit{config['limit']}"
    output_path = output_dir / f"{config['name']}{suffix}.hdf5"
    write_results(
        output_path,
        config,
        test_set,
        coefficients,
        donors,
        permutations,
        distance,
        baselines,
        results,
        checkpoints,
    )
    print(f"Wrote {output_path}")
    print_summary(baselines, results, config["num_other_donors"])
    print_dose_response(distance, results)


if __name__ == "__main__":
    main()
