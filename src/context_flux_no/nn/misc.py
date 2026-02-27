from collections.abc import Callable, Sequence
from functools import partial
from itertools import repeat
from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import pack, unpack
from jaxtyping import Array, Float, Key, Shaped

from ..custom_types import ComplexArray, FloatArray


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


def maybe_split(
    key: Key[Array, ""] | None, num: int = 2
) -> Key[Array, " {num}"] | list[None]:
    """Given a key value that can either be a JAX PRNGKeyArray or None, split it to
    return either the splitted subkeys, or a list of Nones with length equal to the
    number of subkeys requested.

    This is useful in __call__ methods of neural network layers involving sublayers that
    require random keys during training."""
    if key is not None:
        subkeys = jax.random.split(key, num)
    else:
        subkeys = [None] * num
    return subkeys


@partial(jax.jit, static_argnames=["n"])
def atleast_nd(x: Array, n: int):
    """Convert inputs to arrays with at least n dimensions. All new dimensions are
    appended at the end.

    Note that this function behaves differently compared to `jax.numpy.atleast_3d` for
    1D inputs. While `atleast_3d` appends dimensions to before and after the original
    dimension ((N, ) -> (1, N, 1)), atleast_nd will append dimensions at the end
    ((N, ) -> (N, 1, 1))."""
    if x.ndim >= n:
        return x
    else:
        return jnp.expand_dims(x, axis=tuple((range(-1, -(n - x.ndim + 1), -1))))


T = TypeVar("T")


def to_ntuple(x: T | Sequence[T], n: int) -> tuple[T, ...]:
    if isinstance(x, Sequence):
        if len(x) == n:
            return tuple(x)
        else:
            raise ValueError(f"Length of {x} (length = {len(x)} is not equal to {n})")
    else:
        return tuple(repeat(x, n))


def apply_along_axis(
    fn: Callable[[Shaped[Array, " N"]], Shaped[Array, " M"]], x: Array, axis: int = -1
) -> Array:
    """Apply an arary function fn along an axis of the input x. fn *must* map a 1D tensor to a
    1D tensor."""

    if axis >= x.ndim or axis < -x.ndim:
        raise ValueError(
            f"""Invalid axis. Requested to map function over {axis=} for a {x.ndim} 
            array."""
        )

    x: Shaped[Array, "*rest N"] = jnp.swapaxes(x, axis, -1)
    x_flat, shape_info = pack([x], "* C")
    y_flat: Shaped[Array, "*rest M"] = eqx.filter_vmap(fn)(x_flat)
    y = unpack(y_flat, shape_info, "* C")[0]
    return jnp.swapaxes(y, axis, -1)


def apply_componentwise(
    fn: Callable[[FloatArray], FloatArray],
) -> Callable[[ComplexArray], ComplexArray]:
    """Given a function fn: R -> R, return a complex function C->C that applies
    fn separately on the real and imaginary components of the input x.

    This is useful for neural network architectures involving complex numbered data,
    such as the Fourier Neural Operator and its variants"""

    def _inner(x: ComplexArray) -> ComplexArray:
        return jax.lax.complex(fn(jax.lax.real(x)), fn(jax.lax.imag(x)))

    return _inner


def standardize(
    x: Float[Array, " *dims"], axis: int | Sequence[int], eps: float = 1e-7
) -> tuple[Float[Array, " *dims"], dict[str, Float[Array, " *dims"]]]:
    r"""Given an array $x$, return its standardized version
    $\tilde{x}=\frac{x-\mu}{\sigma + \epsilon}$, where $\epsilon$ is a small value used
    to avoid division by zero and mean $\mu$ and standard deviation $\sigma$ are
    calculated along dimension in `dims`. Additionally return a dictionary of
    statistics used for standardization"""
    mu = jnp.mean(x, axis=axis, keepdims=True)
    sigma = jnp.std(x, axis=axis, keepdims=True) + eps

    x_normed = (x - mu) / sigma
    stats = {"mean": mu, "std": sigma}
    return x_normed, stats


def destandardize(
    x: Float[Array, " *dims"], mean: Float[Array, " *dims"], std: Float[Array, " *dims"]
) -> Float[Array, " *dims"]:
    return x * std + mean
