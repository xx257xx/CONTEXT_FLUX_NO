from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import optax
import wandb
from context_flux_no.models.dpot import DPOT
from context_flux_no.training.loader import SegmentLoader
from jaxtyping import Array, Float
from orbax.checkpoint import v1 as ocp


def loss_fn(
    model, batch: tuple[Float[Array, "..."], ...]
) -> tuple[Float[Array, ""], dict]:
    u, dt, dx = batch
    u0, u1 = u[:, :-1], u[:, -1]
    u1_pred = eqx.filter_vmap(model)(u0)[0]
    return jnp.mean((u1 - u1_pred) ** 2), dict()


def train(
    model: DPOT,
    dataloader: SegmentLoader,
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
