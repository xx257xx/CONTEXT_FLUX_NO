import equinox as eqx
import jax
import jax.numpy as jnp
from context_flux_no.nn.misc import apply_along_axis, atleast_nd, maybe_split
from jaxtyping import Array, Key


def test_maybe_split():
    subkeys = maybe_split(jax.random.key(0), 3)
    assert isinstance(subkeys, Key[Array, " 3"])
    assert len(subkeys) == 3

    subkeys_dummy = maybe_split(None, 4)
    assert subkeys_dummy == [None] * 4


def test_atleast_nd():
    x = jnp.ones((2, 5, 4))

    # n < x.ndim
    assert jnp.array_equal(atleast_nd(x, n=2), x)

    # n = x.ndim
    assert jnp.array_equal(atleast_nd(x, n=3), x)

    # n > x.ndim
    assert atleast_nd(x, n=4).shape == x.shape + (1,)
    assert atleast_nd(x, n=5).shape == x.shape + (1, 1)


def test_apply_along_axis():
    test_arr = jax.random.normal(jax.random.key(1), (10, 32, 2))

    fn1 = eqx.nn.Linear(32, 32, key=jax.random.key(0))
    # Check for length preserving function
    assert apply_along_axis(fn1, test_arr, axis=1).shape == (10, 32, 2)

    fn2 = eqx.nn.Linear(10, 32, key=jax.random.key(1))
    # Check for length changing function
    assert apply_along_axis(fn2, test_arr, axis=0).shape == (32, 32, 2)
