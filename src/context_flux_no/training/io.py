from pathlib import Path

import equinox as eqx
import jax
import more_itertools
from hydra.utils import instantiate
from jaxtyping import PyTree
from omegaconf import OmegaConf
from orbax.checkpoint import v1 as ocp


def _infer_step(ckpt_dir: Path) -> int:
    """Given a checkpoint directory created by orbax.checkpoint consisting of a
    checkpoint from a single train step, determine and return the corresponding step
    number."""
    ckpt_step_dirs = filter(
        lambda p: p.is_dir() and p.stem.isnumeric(), ckpt_dir.iterdir()
    )
    try:
        return int(more_itertools.one(ckpt_step_dirs).stem)
    except ValueError:
        raise ValueError(
            """Given directory contains checkpoints from multiple steps, and thus step 
            cannot be inferred."""
        )


def load_model(
    loaddir: str | Path,
    step_number: int | None = None,
) -> PyTree:
    """Loads model from a checkpoint created by the `save_model` function.

    loaddir: Directory containing the checkpoints
    step_number: Number corresponding to the specific checkpoint
    """

    loaddir = Path(loaddir).resolve()
    if step_number is None:
        step_number = _infer_step(loaddir)

    with ocp.training.Checkpointer(loaddir) as mngr_load:
        model_config = mngr_load.root_metadata().custom_metadata
        model_backbone = instantiate(OmegaConf.create(model_config))

        weights_backbone, rest = eqx.partition(
            model_backbone, lambda x: isinstance(x, jax.Array)
        )
        weights_load = mngr_load.load_pytree(step_number, weights_backbone)

    return eqx.combine(weights_load, rest)
