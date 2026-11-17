"""Parameters, FLOPs, memory and run time of the submitted models on one family.

This script answers the cost questions of the ICLR 2027 reviews (rebuttal plan,
action 4b): Reviewer DHwv, W4, on the overhead of the hypernetwork, and Reviewer AyEF,
Q6, on the training and inference cost. A preset holds the test set and the seed-0
checkpoints of the submission:

    python scripts/evaluation/compute_cost.py --preset cubic_1d
    python scripts/evaluation/compute_cost.py --preset euler_2d

Every value of a preset is also a command-line argument, and an argument overrides the
preset. A family without a preset is described by arguments alone, with one
`--checkpoint NAME=PATH` per model:

    python scripts/evaluation/compute_cost.py \
        --name <family> --loader legacy_hdf5 --data-path <test file> \
        --checkpoint-root <checkpoint directory> \
        --checkpoint "DPOT=<run directory>" --checkpoint "DISCO=<run directory>" \
        --eval-batch <batch> --train-batch <batch>

The counts and the held-fixed rollout support the model classes of the 1D families and
of 2D Euler. The 2D cubic HFluxNO of the paper is `HyperFluxFNOLocal2D`, which is not
on this branch, so 2D cubic is not measured (rebuttal plan, 2026-11-17).

Each model is measured in three ways:

- a rollout of `--rollout-steps` steps from the first `--context-length` frames of the
  test trajectories, at batch 1 and at `--eval-batch`;
- for HyperFluxFNOLocal, the same rollout with the flux networks regenerated every
  `--refresh-every` steps (action 3e);
- one training step of the one-step loss with AdamW at `--train-batch`.

Every measurement runs in a fresh process, so that the peak memory of the device
belongs to that measurement alone. The test windows replace the training data in the
training step: its cost does not depend on the data, except for DISCO, whose adaptive
solver takes more steps on harder states.

A preset with `concatenation_config` also measures the context-concatenation variant of
Table 1, built untrained from that configuration, since its cost does not depend on its
weights. If it is faster than HyperFluxFNOLocal at `--eval-batch`, a narrower
concatenation model with the flux network of HyperFluxFNOLocal is measured as well, to
tell whether the width of the variant or the per-sample weights of the hypernetwork
explain the difference. The narrow model is a control and is not reported in the paper.

The script writes every measurement to one HDF5 file and prints a table.
"""

import argparse
import concurrent.futures
import dataclasses
import json
import multiprocessing
import tempfile
from pathlib import Path
from typing import Any

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
from context_flux_no.cost import (
    measure,
    Measurement,
    parameter_breakdown,
    step_flop_breakdown,
)
from context_flux_no.evaluation import rollout_with_refresh
from context_flux_no.models.multiphysics import HyperFluxFNOLocal
from context_flux_no.training.io import load_model
from context_flux_no.training.loss import OneStepLoss
from context_flux_no.training.trainer import Trainer, TrainerState
from hydra.utils import instantiate
from omegaconf import OmegaConf
from rollout_diagnostics import (
    load_legacy_hdf5,
    load_well_zarr,
    PRESETS as DIAGNOSTICS_PRESETS,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONCATENATION = "Concatenation"
CONCATENATION_CONTROL = "Concatenation (width 128, control)"
# The flux network and trunk of HyperFluxFNOLocal on the 1D families (Table 4): a
# trunk of one hidden layer of width 128 and a flux MLP of depth 4 and width 128.
CONTROL_OVERRIDES = {
    "width_hyper": 128,
    "depth_hyper": 1,
    "width_flux": 128,
    "depth": 4,
}
# The settings of the training runs of the submission (Appendix C).
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
DATA_KEYS = ("loader", "data_path", "time_stride", "well_dataset_name")


def _seed_zero(preset: str) -> list[str]:
    """The seed-0 checkpoints of a preset of `rollout_diagnostics.py`."""
    return [e for e in DIAGNOSTICS_PRESETS[preset]["checkpoint"] if "/seed=0/" in e]


def _data_of(preset: str) -> dict[str, Any]:
    """The test set of a preset of `rollout_diagnostics.py`."""
    preset_values = DIAGNOSTICS_PRESETS[preset]
    data = {key: preset_values[key] for key in DATA_KEYS if key in preset_values}
    if "well_exclude_filters" in preset_values:
        data["well_exclude_filters"] = preset_values["well_exclude_filters"]
    return data


# The test sets and checkpoints are those of `rollout_diagnostics.py`. The batch sizes
# are the evaluation batch of that script and the training batch of the submission.
PRESETS: dict[str, dict[str, Any]] = {
    "cubic_1d": {
        **_data_of("cubic_1d"),
        "checkpoint": _seed_zero("cubic_1d"),
        "eval_batch": 2000,
        "train_batch": 256,
        "concatenation_config": "scripts/training/configs/model/fluxno_appended.yaml",
    },
    "euler_2d": {
        **_data_of("euler_2d"),
        "checkpoint": _seed_zero("euler_2d"),
        "eval_batch": 50,
        "train_batch": 64,
    },
}

DEFAULTS: dict[str, Any] = {
    "data_root": "./data",
    "checkpoint_root": "./checkpoints",
    "output_dir": "./data/analysis/compute_cost",
    "time_stride": 1,
    "well_dataset_name": None,
    "well_exclude_filters": [],
    "context_length": 20,
    "rollout_steps": 20,
    "refresh_every": 8,
    "training_steps": 50000,
    "concatenation_config": None,
    "repeats": 20,
    "warmup": 3,
    "gpu": None,
}


@dataclasses.dataclass(frozen=True)
class Task:
    """One measurement, run in its own process.

    **Attributes:**

    - `label`: The name of the model in the output.
    - `checkpoint`: The checkpoint directory, or `None` for a model built untrained.
    - `config`: The resolved model configuration of an untrained model, or `None`.
    - `kind`: `rollout`, `refresh` or `train`.
    - `batch`: The number of rollouts or training samples.
    - `with_counts`: Whether to count parameters and one-step FLOPs as well.
    """

    label: str
    checkpoint: str | None
    config: dict[str, Any] | None
    kind: str
    batch: int
    with_counts: bool


@dataclasses.dataclass(frozen=True)
class Settings:
    """What every task shares: the data file, the model args and the timing setup."""

    data_file: str
    model_args: tuple[float, ...]
    rollout_steps: int
    refresh_every: int
    repeats: int
    warmup: int
    gpu: int | None


def _build_model(task: Task):
    if task.checkpoint is not None:
        return load_model(task.checkpoint)
    return instantiate(OmegaConf.create(task.config))


def run_task(task: Task, settings: Settings) -> dict[str, Any]:
    """Measure one model in one way. Runs in a worker process."""
    if settings.gpu is not None:
        jax.config.update("jax_default_device", jax.devices("gpu")[settings.gpu])
    if task.kind not in ("rollout", "refresh", "train"):
        raise ValueError(f"Unknown kind of measurement {task.kind!r}.")
    model = _build_model(task)
    data = np.load(settings.data_file)
    args = settings.model_args
    timing = dict(repeats=settings.repeats, warmup=settings.warmup)

    if task.kind == "train":
        segments = jnp.asarray(data["train_segments"][: task.batch])
        optimizer = optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # The trainer only builds its checkpoint path here and writes nothing.
        trainer = Trainer(
            optimizer=optimizer,
            loss_fn=OneStepLoss(),
            checkpoint_dir=tempfile.gettempdir(),
            checkpoint_name=None,
            wandb_kwargs={},
            config_dict={},
        )
        state = TrainerState.init(model, optimizer, key=jax.random.key(0))
        # `Trainer.train_step` is `_train_step` under `eqx.filter_jit`, whose compiled
        # program does not expose XLA's cost and memory analysis.
        measurement = measure(trainer._train_step, state, segments, args, **timing)
    else:
        model = eqx.nn.inference_mode(model, True)
        contexts = jnp.asarray(data["contexts"][: task.batch])
        steps, every = settings.rollout_steps, settings.refresh_every
        if task.kind == "rollout":
            one = lambda m, c: m.rollout(c, args, num_steps=steps)[0]  # noqa: E731
        else:
            one = lambda m, c: rollout_with_refresh(m, c, args, steps, every)  # noqa: E731
        measurement = measure(
            lambda m, cs: jax.vmap(lambda c: one(m, c))(cs), model, contexts, **timing
        )

    result: dict[str, Any] = {
        "class": type(model).__name__,
        "device": jax.devices()[settings.gpu or 0].device_kind,
        "measurement": measurement,
    }
    if task.with_counts:
        model = eqx.nn.inference_mode(model, True)
        context = jnp.asarray(data["contexts"][0])
        result["parameters"] = parameter_breakdown(model)
        result["flops"] = step_flop_breakdown(model, context, args)
    return result


def _run(task: Task, settings: Settings) -> dict[str, Any]:
    """Run a task in a fresh process, so that its peak memory is its own."""
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(1, mp_context=context) as pool:
        return pool.submit(run_task, task, settings).result()


def measure_model(
    label: str,
    checkpoint: str | None,
    model_config: dict[str, Any] | None,
    eval_batch: int,
    train_batch: int,
    settings: Settings,
) -> list[tuple[Task, dict[str, Any]]]:
    """Run every measurement of one model, each in its own process.

    The rollout at batch 1 runs first and also counts the parameters and FLOPs. It
    tells the class of the model, which decides whether the rollout with the flux
    networks held fixed is measured.
    """
    first = Task(label, checkpoint, model_config, "rollout", 1, True)
    entries = [(first, _run(first, settings))]
    kinds = [("rollout", eval_batch)]
    if entries[0][1]["class"] == HyperFluxFNOLocal.__name__:
        kinds += [("refresh", 1), ("refresh", eval_batch)]
    kinds += [("train", train_batch)]
    for kind, batch in kinds:
        print(f"{label}: {kind} at batch {batch}")
        task = Task(label, checkpoint, model_config, kind, batch, False)
        entries.append((task, _run(task, settings)))
    return entries


def _concatenation_config(
    path: Path,
    num_spatial_dims: int,
    channels: int,
    context_length: int,
    overrides: dict[str, int],
) -> dict[str, Any]:
    """Resolve a model configuration of the training scripts outside of Hydra, and
    replace the settings in `overrides`."""
    composed = OmegaConf.create(
        {
            "model": OmegaConf.load(path),
            "data": {"num_spatial_dims": num_spatial_dims, "channels": channels},
            "training": {"context_length": context_length},
        }
    )
    model = OmegaConf.to_container(composed, resolve=True)["model"]
    unknown = set(overrides) - set(model)
    if unknown:
        raise ValueError(f"{path} has no settings {sorted(unknown)}.")
    return model | overrides


def _per_sample_ms(measurement: Measurement, batch: int, steps: int) -> float:
    return 1e3 * float(np.median(measurement.seconds)) / batch / steps


def write_results(
    path: Path,
    config: dict[str, Any],
    settings: Settings,
    results: dict[str, list[tuple[Task, dict[str, Any]]]],
) -> None:
    """Write every measurement to one file.

    Each model is a group with the attributes `class` and `checkpoint`, the groups
    `parameters` and `flops` whose attributes are the counts, and one group per
    measurement, named `{kind}_batch{batch}`. A measurement group holds the dataset
    `seconds` and the other fields of `Measurement` as attributes.
    """
    with h5py.File(path, "w") as f:
        f.attrs["config"] = json.dumps(config)
        f.attrs["model_args"] = settings.model_args
        (first_entry, *_), *_ = results.values()
        f.attrs["device"] = first_entry[1]["device"]
        f.attrs["jax_version"] = jax.__version__
        for label, entries in results.items():
            group = f.create_group(f"models/{label}")
            for task, result in entries:
                group.attrs["class"] = result["class"]
                group.attrs["checkpoint"] = str(task.checkpoint)
                for counts in ("parameters", "flops"):
                    if counts in result:
                        group.create_group(counts).attrs.update(result[counts])
                m = result["measurement"]
                sub = group.create_group(f"{task.kind}_batch{task.batch}")
                sub.create_dataset("seconds", data=m.seconds)
                for field in dataclasses.fields(m):
                    if field.name != "seconds":
                        sub.attrs[field.name] = getattr(m, field.name)


def print_summary(
    results: dict[str, list[tuple[Task, dict[str, Any]]]],
    settings: Settings,
    training_steps: int,
) -> None:
    """Print one row per model and per way of rolling it out."""
    steps = settings.rollout_steps
    print(
        f"\n{'model':38s}{'params':>9s}{'generated':>10s}{'GFLOP/step':>11s}"
        f"{'ms/sample':>10s}{'ms (b=1)':>10s}{'peak MiB':>10s}"
        f"{'train ms':>10s}{'GPU-h':>7s}{'peak MiB':>10s}"
    )
    for label, entries in results.items():
        by_kind = {(t.kind, t.batch): r for t, r in entries}
        counts = by_kind[("rollout", 1)]
        params, flops = counts["parameters"], counts["flops"]
        eval_batch = max(b for k, b in by_kind if k == "rollout")
        (train_batch,) = [b for k, b in by_kind if k == "train"]
        train = by_kind[("train", train_batch)]["measurement"]
        train_s = float(np.median(train.seconds))
        for kind in ("rollout", "refresh"):
            if (kind, 1) not in by_kind:
                continue
            m_eval = by_kind[(kind, eval_batch)]["measurement"]
            m_one = by_kind[(kind, 1)]["measurement"]
            if kind == "rollout":
                row = (
                    f"{label:38s}{params['total'] / 1e6:>8.2f}M"
                    + (
                        f"{params['generated'] / 1e6:>9.3f}M"
                        if "generated" in params
                        else f"{'—':>10s}"
                    )
                    + f"{flops['step'] / 1e9:>11.3f}"
                )
            else:
                row = f"{'  flux held ' + str(settings.refresh_every) + ' steps':38s}"
                row += " " * 30
            row += (
                f"{_per_sample_ms(m_eval, eval_batch, steps):>10.4f}"
                f"{_per_sample_ms(m_one, 1, steps):>10.3f}"
                f"{m_eval.peak_bytes / 2**20:>10.0f}"
            )
            if kind == "rollout":
                row += (
                    f"{1e3 * train_s:>10.1f}{train_s * training_steps / 3600:>7.2f}"
                    f"{train.peak_bytes / 2**20:>10.0f}"
                )
            print(row)
        parts = ", ".join(f"{k} {v / 1e9:.4g}" for k, v in flops.items() if k != "step")
        if parts:
            print(f"  GFLOP per part: {parts}")
        named = ", ".join(f"{k} {v:,}" for k, v in params.items() if k != "total")
        if named:
            print(f"  parameters: {named}")
    print(
        f"\nms/sample: per sample and step at the evaluation batch; ms (b=1): per step "
        f"at batch 1; GPU-h: the training step times {training_steps} steps, without "
        f"data loading. GFLOP/step is one step for one sample; for DISCO it counts one "
        f"iteration of its adaptive solver."
    )


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
        help="A checkpoint directory under --checkpoint-root, one per model.",
    )
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--refresh-every", type=int)
    parser.add_argument("--eval-batch", type=int)
    parser.add_argument("--train-batch", type=int)
    parser.add_argument(
        "--training-steps",
        type=int,
        help="The number of training steps that the training time is scaled to.",
    )
    parser.add_argument(
        "--concatenation-config",
        help="A model configuration of the context-concatenation variant, relative "
        "to the repository root. Default: none for a family without a preset.",
    )
    parser.add_argument("--repeats", type=int, help="The number of timed calls.")
    parser.add_argument("--warmup", type=int, help="Untimed calls before them.")
    parser.add_argument("--output-dir")
    parser.add_argument("--gpu", type=int, help="Index of the GPU to run on.")
    given = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}

    preset = PRESETS[given["preset"]] if "preset" in given else {}
    config = DEFAULTS | preset | given
    config.setdefault("name", config.get("preset"))
    required = (
        "name",
        "loader",
        "data_path",
        "checkpoint",
        "eval_batch",
        "train_batch",
    )
    missing = [key for key in required if config.get(key) is None]
    if missing:
        parser.error(f"Without a preset, these arguments are required: {missing}")
    if config["loader"] == "well_zarr" and config["well_dataset_name"] is None:
        parser.error("--loader well_zarr requires --well-dataset-name")
    return config


def main() -> None:
    config = parse_config()
    data_path = Path(config["data_root"]) / config["data_path"]
    num_needed = max(config["eval_batch"], config["train_batch"])
    if config["loader"] == "legacy_hdf5":
        test_set = load_legacy_hdf5(data_path, config["time_stride"], num_needed)
    else:
        test_set = load_well_zarr(
            data_path,
            config["well_dataset_name"],
            config["well_exclude_filters"],
            None,
            num_needed,
        )
    trajectories = test_set.trajectories
    context_length = config["context_length"]
    if trajectories.shape[0] < num_needed:
        raise ValueError(
            f"{trajectories.shape[0]} valid trajectories, but the batches need "
            f"{num_needed}."
        )
    if trajectories.shape[1] < context_length + 1:
        raise ValueError(
            f"Trajectories of {trajectories.shape[1]} frames are too short."
        )
    # The parent process does not touch the device: JAX would reserve most of its
    # memory here, away from the processes that measure.
    print(
        f"{config['name']}: windows of shape {trajectories.shape[1:]}, model args "
        f"{test_set.model_args}"
    )

    checkpoint_root = Path(config["checkpoint_root"])
    models: list[tuple[str, str | None, dict[str, Any] | None]] = []
    for entry in config["checkpoint"]:
        label, _, pattern = entry.partition("=")
        matches = sorted(checkpoint_root.glob(pattern))
        if len(matches) != 1:
            raise ValueError(f"{entry!r} matches {len(matches)} paths, expected one.")
        models.append((label, str(matches[0]), None))
    num_spatial_dims, channels = trajectories.ndim - 3, trajectories.shape[2]
    concatenation_path = config["concatenation_config"]
    if concatenation_path is not None:
        concatenation_path = REPOSITORY_ROOT / concatenation_path
        models.append(
            (
                CONCATENATION,
                None,
                _concatenation_config(
                    concatenation_path, num_spatial_dims, channels, context_length, {}
                ),
            )
        )

    results: dict[str, list[tuple[Task, dict[str, Any]]]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        data_file = str(Path(tmp) / "windows.npz")
        np.savez(
            data_file,
            contexts=trajectories[: config["eval_batch"], :context_length],
            train_segments=trajectories[: config["train_batch"], : context_length + 1],
        )
        settings = Settings(
            data_file=data_file,
            model_args=tuple(test_set.model_args),
            rollout_steps=config["rollout_steps"],
            refresh_every=config["refresh_every"],
            repeats=config["repeats"],
            warmup=config["warmup"],
            gpu=config["gpu"],
        )

        batches = dict(
            eval_batch=config["eval_batch"], train_batch=config["train_batch"]
        )
        for label, checkpoint, model_config in models:
            print(f"{label}: {checkpoint or 'untrained, from its configuration'}")
            results[label] = measure_model(
                label, checkpoint, model_config, settings=settings, **batches
            )

        if concatenation_path is not None and "HyperFluxFNOLocal" in results:

            def eval_ms(label):
                ((task, result),) = [
                    (t, r)
                    for t, r in results[label]
                    if (t.kind, t.batch) == ("rollout", config["eval_batch"])
                ]
                return _per_sample_ms(
                    result["measurement"], task.batch, settings.rollout_steps
                )

            if eval_ms(CONCATENATION) < eval_ms("HyperFluxFNOLocal"):
                print(
                    "Concatenation is faster than HyperFluxFNOLocal: measuring the "
                    "control."
                )
                control = _concatenation_config(
                    concatenation_path,
                    num_spatial_dims,
                    channels,
                    context_length,
                    CONTROL_OVERRIDES,
                )
                results[CONCATENATION_CONTROL] = measure_model(
                    CONCATENATION_CONTROL, None, control, settings=settings, **batches
                )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config['name']}.hdf5"
    write_results(output_path, config, settings, results)
    print(f"Wrote {output_path}")
    print_summary(results, settings, config["training_steps"])


if __name__ == "__main__":
    main()
