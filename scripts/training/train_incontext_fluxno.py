from pathlib import Path
from typing import Any

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import optax
import wandb
import xarray as xr
from context_flux_no.models.hyperfluxfno import ViTContextHyperFluxFNO
from context_flux_no.training import PDEDataset
from context_flux_no.training.loader import ContextSegmentLoader
from jaxtyping import Array, Float
from omegaconf import DictConfig, OmegaConf
from orbax.checkpoint import v1 as ocp


def loss_fn(
    model: ContextSegmentLoader, batch: tuple[Float[Array, "..."]]
) -> tuple[Float[Array, ""], dict]:
    context, u, dt, dx = batch
    u0, u1 = u[:, 0], u[:, 1]
    u1_pred = eqx.filter_vmap(
        # For non-zero dropout, key value may become important
        lambda context_, u0_: model(context_, u0_, dt, dx, key=jax.random.key(0))
    )(context, u0)
    loss_train = jnp.mean((u1 - u1_pred) ** 2)
    return loss_train, None


def train(
    model: ViTContextHyperFluxFNO,
    dataloader: ContextSegmentLoader,
    optimizer: optax.GradientTransformation,
    loss_fn,
    max_steps: int,
    checkpoint_path: str | Path,
    checkpoint_name: str,
    config_dict: dict[str, Any],
    wandb_entity: str,
    wandb_project: str,
):
    max_steps = int(max_steps)
    loss_grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)

    @eqx.filter_jit
    def train_step(model_, loader_state_, opt_state_):
        batch, loader_state_next = dataloader.load_batch(loader_state_)
        (loss, aux), grads = loss_grad_fn(model_, batch)
        updates, opt_state_next = optimizer.update(
            grads, opt_state_, eqx.filter(model_, eqx.is_array)
        )
        model_ = eqx.apply_updates(model_, updates)
        return loss, aux, model_, loader_state_next, opt_state_next

    loader_state = dataloader.init()
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    savepath = Path(checkpoint_path) / checkpoint_name
    savepath = savepath.resolve()
    savepath.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.training.Checkpointer(
        savepath,
        preservation_policy=ocp.training.preservation_policies.BestN(
            get_metric_fn=lambda metrics: metrics["train_loss"],
            reverse=True,
            n=1,
        ),
        custom_metadata=config_dict["model"],
    )
    logger = wandb.init(entity=wandb_entity, project=wandb_project, config=config_dict)

    with checkpointer as ckptr:
        loss_history = []
        for step in range(max_steps):
            loss, aux, model_next, loader_state, opt_state = train_step(
                model, loader_state, opt_state
            )
            loss_scalar = loss.item()
            metrics = {"train_loss": loss_scalar}
            logger.log(metrics, step=step)

            print(f"Step: {step}: loss = {loss_scalar}")
            loss_history.append(loss_scalar)

            ckptr.save_pytree(
                step,
                eqx.filter(model, eqx.is_array),
                metrics=metrics,
            )
            model = model_next
    return model, jnp.asarray(loss_history)


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    data_train, data_validate = (
        PDEDataset.from_xarray(xr.load_dataset(cfg.data.loadpath))
        .downsample_time(cfg.data.downsample_time)
        .split_by_time(cfg.data.time_split_index)
    )

    model = hydra.utils.instantiate(cfg.model)
    loader_train = ContextSegmentLoader(
        data_train,
        cfg.training.context_size,
        cfg.training.segment_length,
        cfg.training.batch_size,
    )
    optimizer = hydra.utils.instantiate(cfg.training.optimizer)

    model_trained = train(
        model=model,
        dataloader=loader_train,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_steps=cfg.training.max_steps,
        checkpoint_path=cfg.training.checkpoint_dir,
        checkpoint_name=cfg.training.checkpoint_name,
        config_dict=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        wandb_entity=cfg.wandb.entity,
        wandb_project=cfg.wandb.project,
    )
    return model_trained


if __name__ == "__main__":
    main()
