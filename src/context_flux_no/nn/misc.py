from collections.abc import Sequence
from itertools import repeat
from typing import TypeVar

import jax.numpy as jnp


def to_complex_dtype(dtype):
    """Convert a given jax.numpy dtype into a corresponding complex dtype.
    Currently, only supports jnp.float32 and jnp.float64."""
    if dtype == jnp.float32:
        return jnp.complex64
    elif dtype == jnp.float64:
        return jnp.complex128
    else:
        raise ValueError(
            f"Conversion of {dtype} into a complex dtype is not supported."
        )


T = TypeVar("T")


def to_ntuple(x: T | Sequence[T], n: int) -> tuple[T, ...]:
    if isinstance(x, Sequence):
        if len(x) == n:
            return tuple(x)
        else:
            raise ValueError(f"Length of {x} (length = {len(x)} is not equal to {n})")
    else:
        return tuple(repeat(x, n))
