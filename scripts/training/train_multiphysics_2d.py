from pathlib import Path
from typing import Optional

import hydra
import jax
import xarray as xr
from context_flux_no.models.multiphysics import (
    AbstractMultiphysicsOperator,
    HyperFluxFNO,
    HyperFluxFNOLocal,
)
from context_flux_no.training.loader import SegmentLoaderBackground
from context_flux_no.training.loss import PushforwardOneStepLoss
from context_flux_no.training.trainer import Trainer
from omegaconf import DictConfig, OmegaConf


def get_coordinate_spacing(dataset: xr.Dataset, coordinate: str) -> float:
    """Return the uniform spacing of a dataset coordinate."""
    if coordinate not in dataset.coords:
        raise ValueError(
            f"Dataset does not contain the required coordinate '{coordinate}'. "
            f"Available coordinates: {list(dataset.coords)}"
        )

    values = dataset[coordinate]

    if values.size < 2:
        raise ValueError(
            f"Coordinate '{coordinate}' must contain at least two points."
        )

    return float(values[1] - values[0])


def validate_spatial_2d_dataset(dataset: xr.Dataset) -> None:
    """Validate that the dataset has time and two spatial coordinates."""
    required_coordinates = ("t", "x", "y")
    missing = [
        coordinate
        for coordinate in required_coordinates
        if coordinate not in dataset.coords
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing required coordinates: {missing}. "
            f"Available coordinates: {list(dataset.coords)}"
        )


def get_loss_args(
    model: AbstractMultiphysicsOperator,
    dataset: xr.Dataset,
) -> Optional[tuple[float, float, float]]:
    """
    Return time and 2D spatial grid spacing for flux-based models.

    The returned tuple is:
        (dt, dx, dy)
    """
    if isinstance(model, (HyperFluxFNO, HyperFluxFNOLocal)):
        validate_spatial_2d_dataset(dataset)

        dt = get_coordinate_spacing(dataset, "t")
        dx = get_coordinate_spacing(dataset, "x")
        dy = get_coordinate_spacing(dataset, "y")

        return dt, dx, dy

    return None


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Manually select the GPU to run on.
    if cfg.gpu_id != "auto":
        gpu_id = int(cfg.gpu_id)
        gpu_devices = jax.devices("gpu")

        if gpu_id < 0 or gpu_id >= len(gpu_devices):
            raise ValueError(
                f"Invalid gpu_id={gpu_id}. "
                f"Available GPU indices: 0-{len(gpu_devices) - 1}"
            )

        jax.config.update("jax_default_device", gpu_devices[gpu_id])

    model = hydra.utils.instantiate(cfg.model)

    dataset_train = (
        xr.open_dataset(
            cfg.data.loadpath_train,
            engine="h5netcdf",
            chunks={},
        )
        .isel(t=slice(0, cfg.data.max_train_time_index))
    )

    dataset_valid = (
        xr.open_dataset(
            cfg.data.loadpath_valid,
            engine="h5netcdf",
            chunks={},
        )
        .isel(t=slice(0, cfg.data.max_train_time_index))
    )

    validate_spatial_2d_dataset(dataset_train)
    validate_spatial_2d_dataset(dataset_valid)

    loss_fn = hydra.utils.instantiate(cfg.loss_fn)

    if isinstance(loss_fn, PushforwardOneStepLoss):
        # Context frames + one pushforward prediction frame + one target frame.
        segment_length = cfg.training.context_length + 2
    else:
        # Context frames + one target frame.
        segment_length = cfg.training.context_length + 1

    loader_train = SegmentLoaderBackground(
        dataset_train,
        segment_length,
        cfg.training.batch_size,
        cfg.training.batches_per_load,
        cfg.training.queue_capacity,
    )

    loader_valid = SegmentLoaderBackground(
        dataset_valid,
        segment_length,
        cfg.training.batch_size,
        cfg.training.batches_per_load,
        cfg.training.queue_capacity,
    )

    trainer = Trainer(
        hydra.utils.instantiate(cfg.training.optimizer),
        loss_fn,
        Path(cfg.training.checkpoint_dir) / cfg.data.name,
        checkpoint_name=(
            Path(model.__class__.__name__)
            / loss_fn.__class__.__name__
            / f"seed={cfg.model.key.seed}"
        ),
        wandb_kwargs=cfg.wandb_kwargs,
        config_dict=OmegaConf.to_container(
            cfg,
            resolve=True,
            throw_on_missing=True,
        ),  # ty: ignore[invalid-argument-type]
    )

    trainer.train(
        model=model,
        train_dataloader=loader_train,
        validation_dataloader=loader_valid,
        loss_args=get_loss_args(model, dataset_train),
        num_steps=cfg.training.max_steps,
    )


if __name__ == "__main__":
    main()