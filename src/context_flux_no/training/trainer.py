import datetime
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack
from dataclasses import replace
from functools import cached_property
from itertools import repeat
from pathlib import Path
from typing import Any, Self, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import wandb
from jaxtyping import Array, PRNGKeyArray, PyTree
from orbax.checkpoint import v1 as ocp

from context_flux_no.custom_types import FilterSpec, FloatScalar, IntScalar


# Inspired by levanter
class TrainerState[M: eqx.Module](eqx.Module):
    step: IntScalar
    model: M
    opt_state: optax.OptState
    training_key: PRNGKeyArray

    optimizer: optax.GradientTransformation = eqx.field(static=True)
    is_trainable: FilterSpec = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        model: M,
        optimizer: optax.GradientTransformation,
        is_trainable: FilterSpec = eqx.is_inexact_array,
        *,
        key: PRNGKeyArray,
    ):
        # Making this as the __init__ will clash with the use of replace()
        opt_state = optimizer.init(
            eqx.filter(model, is_trainable)  # ty: ignore[invalid-argument-type]
        )

        return cls(
            step=jnp.asarray(0, dtype=int),
            model=model,
            opt_state=opt_state,
            training_key=key,
            optimizer=optimizer,
            is_trainable=is_trainable,
        )

    def take_step(
        self,
        grads: M,
    ) -> Self:
        """Given a pytree of model gradients, update model parameters using the
        appropriate optimizer update function."""
        grads = self.filter_trainable(grads)
        updates, opt_state_next = self.optimizer.update(
            grads,  # ty: ignore[invalid-argument-type]
            self.opt_state,
            self.filter_trainable(self.model),  # ty: ignore[invalid-argument-type]
        )
        model_next = eqx.apply_updates(self.model, updates)
        return replace(
            self,
            step=self.step + 1,
            model=model_next,
            opt_state=opt_state_next,
            training_key=jax.random.fold_in(self.training_key, self.step),
        )

    def filter_trainable(self, pytree: M) -> M:
        """Filter the given pytree to retain the trainable model parameters, as
        determined by self.is_trainable. Note that this function works on all pytrees
        that are compatible with self.model."""
        return eqx.filter(pytree, self.is_trainable)


class StepOutput[M: eqx.Module](eqx.Module):
    """Class representing the output of a training step."""

    trainer_state: TrainerState[M]
    loss_train: FloatScalar
    metrics_train: dict[str, Array]
    loss_valid: FloatScalar | None = None
    metrics_valid: dict[str, Array] | None = None

    @property
    def step(self) -> int:
        return int(self.trainer_state.step)

    @property
    def metrics(self) -> dict[str, Array]:
        metrics = {"train_loss": self.loss_train} | {
            f"{k}_train": v for k, v in self.metrics_train.items()
        }
        if self.loss_valid is not None:
            metrics = metrics | {"valid_loss": self.loss_valid}
        if self.metrics_valid is not None:
            metrics = metrics | {f"{k}_valid": v for k, v in self.metrics_valid.items()}
        return jax.tree.map(lambda x: float(x), metrics)

    def maybe_save_model_weights(self, ckptr: ocp.training.Checkpointer):
        weights = eqx.filter(self.trainer_state.model, eqx.is_array)
        ckptr.save_pytree(step=self.step, pytree=weights, metrics=self.metrics)

    def log_metrics(self, logger: wandb.Run):
        # In the future, could change to accept a list of different loggers...
        logger.log(self.metrics, step=self.step)
        print(
            f"""Step: {self.step} | Train loss: {self.loss_train} | Valid loss: 
            {self.loss_valid}"""
        )


Batch = PyTree
M = TypeVar("M", bound=eqx.Module)


class Trainer:
    loss_fn: Callable
    optimizer: optax.GradientTransformation
    checkpoint_path: Path
    wandb_kwargs: dict[str, Any]

    def __init__(
        self,
        optimizer: optax.GradientTransformation,
        loss_fn: Callable,
        checkpoint_dir: str | Path,
        checkpoint_name: str | None,
        wandb_kwargs: dict[str, Any],
        config_dict: dict[str, Any],
    ):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.checkpoint_path = self._make_checkpoint_path(
            checkpoint_dir, checkpoint_name
        )
        self.wandb_kwargs = wandb_kwargs
        self.config_dict = config_dict

    def _make_checkpoint_path(
        self, checkpoint_dir: str | Path, checkpoint_name: str | None
    ) -> Path:
        """Given checkpoint directory and name, return the absolute path to save
        checkpoints in."""
        now = datetime.datetime.now()
        now_str = now.strftime("%y-%m-%d-%H:%M:%S")
        if checkpoint_name is None:
            checkpoint_name = ""

        checkpoint_path = Path(checkpoint_dir) / checkpoint_name / now_str
        # orbax.checkpoint does not like relative paths
        checkpoint_path = checkpoint_path.resolve()
        return checkpoint_path

    def train(
        self,
        model: M,
        train_dataloader: Iterable[Batch],
        validation_dataloader: Iterable[Batch] | None = None,
        loss_args: Any = None,
        *,
        trainable_filterspec: FilterSpec = eqx.is_inexact_array,
        num_steps: int = 5000,
        seed: int = 0,
    ):
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        state = TrainerState[M].init(
            model, self.optimizer, trainable_filterspec, key=jax.random.key(seed)
        )
        if validation_dataloader is None:
            validation_dataloader = repeat(None)
            save_metric = "train_loss"
        else:
            save_metric = "valid_loss"

        with ExitStack() as stack:
            logger = stack.enter_context(
                wandb.init(config=self.config_dict, **self.wandb_kwargs)
            )
            ckptr = stack.enter_context(
                ocp.training.Checkpointer(
                    self.checkpoint_path,
                    preservation_policy=ocp.training.preservation_policies.BestN(
                        get_metric_fn=lambda metrics: metrics[save_metric],
                        reverse=True,
                        n=1,
                    ),
                    custom_metadata=self.config_dict["model"],
                )
            )  # add preservation policy, custom_metadata

            for step_output in self.steps(
                state,
                train_dataloader,
                validation_dataloader,
                loss_args,
                num_steps=num_steps,
            ):
                step_output.log_metrics(logger)
                step_output.maybe_save_model_weights(ckptr)

    def steps(
        self,
        state: TrainerState[M],
        train_dataloader: Iterable[Batch],
        validation_dataloader: Iterable[Batch | None],
        loss_args: Any = None,
        *,
        num_steps: int,
    ) -> Iterator[StepOutput[M]]:
        """A generator function that yields StepOutput instances corresponding to each
        training step."""
        batch_iterator = iter(zip(train_dataloader, validation_dataloader))

        try:
            while int(state.step) < num_steps:
                try:
                    batches: tuple[Batch, Batch | None] = next(batch_iterator)
                except StopIteration:
                    # Train and/or validation dataloader exhausted
                    break
                output = self.train_step(state, *batches, loss_args)
                state = output.trainer_state
                yield output
        finally:
            del batch_iterator

    @cached_property
    def train_step(
        self,
    ) -> Callable[[TrainerState[M], Batch, Batch | None, Any], StepOutput[M]]:
        return eqx.filter_jit(self._train_step)

    def _train_step(
        self,
        state: TrainerState[M],
        batch_train: Batch,
        batch_validation: Batch | None,
        args,
    ) -> StepOutput[M]:
        model = eqx.nn.inference_mode(state.model, False)

        loss_grad_fn = eqx.filter_value_and_grad(self.loss_fn, has_aux=True)
        (loss_train, metrics_train), grads = loss_grad_fn(
            model, batch_train, args, state.training_key
        )
        # metrics_train = metrics_train | {
        #     "norm_weights": jnp.linalg.norm(
        #         jax.flatten_util.ravel_pytree(eqx.filter(model, eqx.is_inexact_array))[
        #             0
        #         ]
        #     ),
        #     "norm_grads": jnp.linalg.norm(
        #         jax.flatten_util.ravel_pytree(eqx.filter(grads, eqx.is_inexact_array))[
        #             0
        #         ]
        #     ),
        #     "norm_batch": jnp.linalg.norm(jnp.reshape(batch_train, shape=(-1))),
        # }
        state_next = state.take_step(grads)

        # If validation batch is given, run model in inference mode
        if batch_validation is not None:
            model_valid = eqx.nn.inference_mode(state.model, True)

            loss_valid, metrics_valid = self.loss_fn(
                model_valid,
                batch_validation,
                args,
                jax.random.fold_in(state.training_key, 1),
            )
        else:
            loss_valid, metrics_valid = None, None
        return StepOutput(
            state_next, loss_train, metrics_train, loss_valid, metrics_valid
        )
