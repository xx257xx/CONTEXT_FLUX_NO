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
