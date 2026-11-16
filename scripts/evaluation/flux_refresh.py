"""Rollout errors of HFluxNO when its flux networks are regenerated every n steps.

This script repeats, on a 1D test set, the experiment of the response to Reviewer AyEF
of 2026-11-12, which was run on 2D cubic (ICLR 2027 rebuttal plan, action 3e). In the
rollout of the submission the encoder reads the last context frames at every step and
the hypernetwork generates new flux networks from them. Here the flux networks are
generated every n steps and held in between. n = 1 is the rollout of the submission,
and n equal to the number of rollout steps generates the flux networks once.

    python scripts/evaluation/flux_refresh.py --preset cubic_1d

Every value of the preset is also a command-line argument, and an argument overrides
the preset. `--limit N` restricts a run to the first N trajectories and writes to a
separate file, for a trial.

The script writes the errors of every segment to one HDF5 file and prints their means.
It reports the 20-step errors of the paper and the errors of the last predicted frame,
which the posted response reports.

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
from context_flux_no.evaluation import (
    evaluate_refresh_rollouts,
    evaluate_segment_baselines,
    evaluate_segments,
    RefreshErrors,
)
from context_flux_no.training.io import load_model
from rollout_diagnostics import load_legacy_hdf5, LoadedTestSet, resolve_checkpoints


# The number of trajectories on which n = 1 is compared with the rollout of the model.
NUM_CHECKED_TRAJECTORIES = 100

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
    "output_dir": "./data/analysis/flux_refresh",
    "time_stride": 1,
    "context_length": 20,
    "rollout_steps": 20,
    "refresh_every": [1, 2, 4, 8, 20],
    "limit": None,
    "chunk_size": 10000,
    "batch_size": 100,
    "gpu": None,
}


def write_results(
    path: Path,
    config: dict[str, Any],
    test_set: LoadedTestSet,
    identity_l2_rollout: np.ndarray,
    results: dict[str, list[RefreshErrors]],
    checkpoints: dict[str, list[Path]],
) -> None:
    """Write the per-segment errors of every model to one file.

    Each model is a group whose datasets have the shape
    `(seeds, trajectories, windows, refresh intervals)`.
    """
    with h5py.File(path, "w") as f:
        f.attrs["config"] = json.dumps(config)
        f.attrs["model_args"] = test_set.model_args
        f.attrs["num_trajectories_in_file"] = test_set.num_in_file
        f.attrs["window_starts"] = config["window_starts"]
        f.attrs["refresh_every"] = config["refresh_every"]
        f.create_dataset("trajectory_indices", data=test_set.trajectory_indices)
        f.create_dataset("baselines/identity/l2_rollout", data=identity_l2_rollout)
        for name, seeds in results.items():
            group = f.create_group(f"models/{name}")
            group.attrs["checkpoints"] = [str(p) for p in checkpoints[name]]
            for field in dataclasses.fields(RefreshErrors):
                group.create_dataset(
                    field.name, data=np.stack([getattr(e, field.name) for e in seeds])
                )


def print_summary(
    results: dict[str, list[RefreshErrors]], refresh_every: list[int]
) -> None:
    """Print the mean of every error at every refresh interval.

    The mean is taken over the segments with a finite error, per seed, and a cell
    shows the mean and the standard deviation over the seeds. The fraction of segments
    left out is printed when it is not zero.
    """
    fields = [field.name for field in dataclasses.fields(RefreshErrors)]
    for name, seeds in results.items():
        print(f"\n{name}, mean over segments")
        print(f"{'refresh every':>14s}" + "".join(f"{field:>22s}" for field in fields))
        for i, interval in enumerate(refresh_every):
            cells, left_out = "", 0.0
            for field in fields:
                values = np.stack([getattr(e, field)[..., i] for e in seeds])
                is_finite = np.isfinite(values)
                per_seed = np.where(is_finite, values, 0.0).sum(
                    axis=(1, 2)
                ) / is_finite.sum(axis=(1, 2))
                cells += f"{per_seed.mean():>12.3e} ± {per_seed.std():.1e}"
                left_out = max(left_out, 1 - is_finite.mean())
            print(f"{interval:>14d}{cells}")
            if left_out:
                print(f"{'':14s}fraction of segments left out: {left_out:.2e}")


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
        "--refresh-every",
        type=int,
        nargs="+",
        help="The numbers of steps for which one set of flux networks is used.",
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
    window_starts = tuple(config["window_starts"])
    refresh_every = list(config["refresh_every"])
    segments = dict(
        context_length=config["context_length"], rollout_steps=config["rollout_steps"]
    )
    print(
        f"{config['name']}: {trajectories.shape[0]} trajectories of shape "
        f"{trajectories.shape[1:]}, {len(window_starts)} window starts, refresh "
        f"every {refresh_every} steps, model args {test_set.model_args}"
    )

    checkpoints = resolve_checkpoints(
        config["checkpoint"], Path(config["checkpoint_root"])
    )
    baselines = evaluate_segment_baselines(
        trajectories, window_starts, chunk_size=config["chunk_size"], **segments
    )
    results: dict[str, list[RefreshErrors]] = {}
    for name, paths in checkpoints.items():
        results[name] = []
        for path in paths:
            print(f"{name}: {path}")
            model = eqx.nn.inference_mode(load_model(path), True)
            errors = evaluate_refresh_rollouts(
                model,
                trajectories,
                window_starts,
                test_set.model_args,
                refresh_every=tuple(refresh_every),
                chunk_size=config["chunk_size"],
                batch_size=config["batch_size"],
                **segments,
            )
            results[name].append(errors)
            if 1 in refresh_every:
                checked = evaluate_segments(
                    model,
                    trajectories[:NUM_CHECKED_TRAJECTORIES],
                    window_starts,
                    test_set.model_args,
                    chunk_size=config["chunk_size"],
                    batch_size=config["batch_size"],
                    **segments,
                )
                refreshed = errors.l2_rollout[
                    :NUM_CHECKED_TRAJECTORIES, :, refresh_every.index(1)
                ]
                with np.errstate(invalid="ignore"):
                    difference = np.nanmax(np.abs(refreshed - checked.l2_rollout))
                print(
                    f"  refresh every 1 step against the rollout of the model, largest "
                    f"difference in l2_rollout on {len(refreshed)} trajectories: "
                    f"{difference:.2e}"
                )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if config["limit"] is None else f"_limit{config['limit']}"
    output_path = output_dir / f"{config['name']}{suffix}.hdf5"
    write_results(
        output_path,
        config,
        test_set,
        baselines.identity.l2_rollout,
        results,
        checkpoints,
    )
    print(f"Wrote {output_path}")
    print_summary(results, refresh_every)


if __name__ == "__main__":
    main()
