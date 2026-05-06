from operator import add
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def is_floating_array(element: Any) -> bool:
    """Returns `True` if `element` is a floating JAX/NumPy array.

    A minor modification of the `equinox.is_inexact_array` implementation."""
    if isinstance(element, (np.ndarray, np.generic)):
        return bool(np.issubdtype(element.dtype, np.floating))
    elif isinstance(element, jax.Array):
        return jnp.issubdtype(element.dtype, jnp.floating)
    else:
        return False


def num_parameters(model: eqx.Module) -> int:
    """Returns the number of trainable parameters in an equinox Module.

    All inexact array elements in non-static leafs of the model PyTree is considered
    to be trainable parameters."""

    model_params = eqx.filter(model, eqx.is_inexact_array)
    return jax.tree.reduce(add, jax.tree.map(jnp.size, model_params), initializer=0)
