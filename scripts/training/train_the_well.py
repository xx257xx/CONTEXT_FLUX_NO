from pathlib import Path

import grain
import hydra
import jax
from context_flux_no.data import TheWellDataSource
from context_flux_no.training.loss import PushforwardOneStepLoss
from context_flux_no.training.trainer import Trainer
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Manually select gpu to run on
    if cfg.gpu_id != "auto":
        jax.config.update("jax_default_device", jax.devices("gpu")[cfg.gpu_id])

    model = hydra.utils.instantiate(cfg.model)

    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    if isinstance(loss_fn, PushforwardOneStepLoss):
        segment_length = cfg.training.context_length + 2
    else:
        segment_length = cfg.training.context_length + 1

    source_train = TheWellDataSource(
        cfg.data.well_base_path,
        cfg.data.well_dataset_name,
        "train",
        window_size=segment_length,
        downsample_spatial=cfg.data.downsample_spatial,
        exclude_field_names=cfg.data.exclude_field_names,
    )
    loader_train = grain.DataLoader(
        data_source=source_train,
        sampler=grain.samplers.IndexSampler(len(source_train), shuffle=True, seed=0),
        operations=[grain.transforms.Batch(batch_size=cfg.training.batch_size)],
        worker_count=cfg.data.worker_count,
    )

    source_valid = TheWellDataSource(
        cfg.data.well_base_path,
        cfg.data.well_dataset_name,
        "valid",
        window_size=segment_length,
        downsample_spatial=cfg.data.downsample_spatial,
        exclude_field_names=cfg.data.exclude_field_names,
    )
    loader_valid = grain.DataLoader(
        data_source=source_valid,
        sampler=grain.samplers.IndexSampler(len(source_valid), shuffle=True, seed=1),
        operations=[grain.transforms.Batch(batch_size=cfg.training.batch_size)],
        worker_count=cfg.data.worker_count,
    )

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
        loss_args=(0.015, 1 / 64, 1 / 64),
        num_steps=cfg.training.max_steps,
    )


if __name__ == "__main__":
    main()
