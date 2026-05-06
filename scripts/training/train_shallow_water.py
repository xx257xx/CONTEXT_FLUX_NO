from pathlib import Path

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


def get_loss_args(model: AbstractMultiphysicsOperator, dataset: xr.Dataset):
    if isinstance(model, (HyperFluxFNO, HyperFluxFNOLocal)):
        dt = float(dataset["t"][1] - dataset["t"][0])
        dx = float(dataset["x"][1] - dataset["x"][0])
        return (dt, dx)
    else:
        return None


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Manually select gpu to run on
    if cfg.gpu_id != "auto":
        jax.config.update("jax_default_device", jax.devices("gpu")[cfg.gpu_id])

    model = hydra.utils.instantiate(cfg.model)

    dataset_train = xr.open_mfdataset(
        sorted(list(Path(cfg.data.loadpath_train).glob("*.hdf5"))),
        combine="nested",
        concat_dim="ic",
        engine="h5netcdf",
    ).isel({"t": slice(0, cfg.data.max_train_time_index)})
    dataset_valid = xr.open_dataset(cfg.data.loadpath_valid, engine="h5netcdf").isel(
        {"t": slice(0, cfg.data.max_train_time_index)}
    )

    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    if isinstance(loss_fn, PushforwardOneStepLoss):
        segment_length = cfg.training.context_length + 2
    else:
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

    # segment_length for loader must be cfg.training.context_length+2 for pushforward
    trainer = Trainer(
        hydra.utils.instantiate(cfg.training.optimizer),
        loss_fn,
        Path(cfg.training.checkpoint_dir) / cfg.data.name,
        checkpoint_name=Path(model.__class__.__name__)
        / loss_fn.__class__.__name__
        / f"seed={cfg.model.key.seed}",
        wandb_kwargs=cfg.wandb_kwargs,
        config_dict=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # ty: ignore[invalid-argument-type],
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
