from operator import add

import equinox as eqx
import jax
import jax.numpy as jnp


def num_parameters(model: eqx.Module) -> int:
    """Returns the number of trainable parameters in an equinox Module.

    All inexact array elements in non-static leafs of the model PyTree is considered
    to be trainable parameters."""

    model_params = eqx.filter(model, eqx.is_inexact_array)
    return jax.tree.reduce(add, jax.tree.map(jnp.size, model_params))
