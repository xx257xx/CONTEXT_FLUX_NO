import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import xarray as xr
from context_flux_no.models.multiphysics import (
    AbstractMultiphysicsOperator,
    HyperFluxFNO,
    HyperFluxFNOLocal,
)
from context_flux_no.training.loader import SegmentLoaderBackground
from context_flux_no.training.loss import PushforwardOneStepLoss
from context_flux_no.training.trainer import Trainer
from jaxtyping import Array, Float, PRNGKeyArray
from omegaconf import DictConfig, OmegaConf


def loss_fn(
    model: AbstractMultiphysicsOperator,
    u: Float[Array, "batch time dim ..."],
    args,
    key: PRNGKeyArray,
) -> tuple[Float[Array, ""], dict]:
    u0, u1 = u[:, :-1], u[:, -1]
    keys = jax.random.split(key, u0.shape[0])
    u1_pred: Float[Array, "batch dim ..."] = eqx.filter_vmap(
        lambda u_, key_: model(u_, args, key=key_)
    )(u0, keys)[0]
    return jnp.mean((u1 - u1_pred) ** 2), dict()


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
    dataset = xr.open_dataset(cfg.data.loadpath, engine="h5netcdf", chunks={}).isel(
        {"t": slice(0, cfg.data.max_train_time_index)}
    )

    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    if isinstance(loss_fn, PushforwardOneStepLoss):
        segment_length = cfg.training.context_length + 2
    else:
        segment_length = cfg.training.context_length + 1

    loader = SegmentLoaderBackground(
        dataset,
        segment_length,
        cfg.training.batch_size,
        cfg.training.batches_per_load,
        cfg.training.queue_capacity,
    )
    # Change to use loss from context_flux_no.training.loss
    # segment_length for loader must be cfg.training.context_length+2 for pushforward
    trainer = Trainer(
        hydra.utils.instantiate(cfg.training.optimizer),
        loss_fn,
        cfg.training.checkpoint_dir,
        checkpoint_name=model.__class__.__name__,
        wandb_kwargs=cfg.wandb_kwargs,
        config_dict=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # ty: ignore[invalid-argument-type],
    )

    trainer.train(
        model=model,
        train_dataloader=loader,
        validation_dataloader=None,
        loss_args=get_loss_args(model, dataset),
        num_steps=cfg.training.max_steps,
    )


if __name__ == "__main__":
    main()
