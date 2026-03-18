import equinox as eqx
import jax
import jax.numpy as jnp
from equinox._misc import default_floating_dtype
from jaxtyping import Array, Float, PRNGKeyArray

from .structured_linear import BlockDiagonalLinear


def linear_scan(
    x: Float[Array, "seq dim"],
    a: Float[Array, "seq dim"],
    h0: Float[Array, " dim"] | None = None,
    *,
    unroll: int,
) -> tuple[Float[Array, "seq dim"], Float[Array, " dim"]]:
    def body_fn(h_prev, inputs_):
        x_t, a_t = inputs_
        h_t = a_t * h_prev + x_t
        return h_t, h_t

    h0 = jnp.zeros_like(x[0]) if h0 is None else h0
    h_last, y = jax.lax.scan(body_fn, h0, (x, a), unroll=unroll)
    return y, h_last


class RG_LRU(eqx.Module):
    """Implementation of the Real-Gated Linear Recurrent Unit from [1], adapted from the
     original implementation in [2].

    This module implements only a subset of the original functionality.
    - Support for reduced precision dropped
    - Support for concatenation of multiple sequences (implemented via pos argument
        and the reset logic in the original code) dropped
    - Only real recurrence weight (a) supported. This is in line with reports of [1]
        saying that complex weights did not yield performance improvement for language
        modeling. TRecViT [3], which is a video model based on this layer also uses real
         weights only.

    [1] S. De et al. Griffin: Mixing Gated Linear Recurrences with Local Attention for
        Efficient Language Models. arXiv:2402.19427 (2024).
    [2] https://github.com/google-deepmind/recurrentgemma/blob/main/recurrentgemma/jax/layers.py#L217
    [3] V. Patraucean et al. TRecViT: A Recurrent Video Transformer. TMLR (2025).
    """

    input_gate: BlockDiagonalLinear
    recurrence_gate: BlockDiagonalLinear
    a: Float[Array, " channels"]

    channels: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        num_heads: int,
        a_init_minval: float = 0.9,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, 3)
        dtype = default_floating_dtype() if dtype is None else dtype

        self.input_gate = BlockDiagonalLinear(
            in_features=channels,
            out_features=channels,
            num_blocks=num_heads,
            dtype=dtype,
            key=keys[0],
        )
        self.recurrence_gate = BlockDiagonalLinear(
            in_features=channels,
            out_features=channels,
            num_blocks=num_heads,
            dtype=dtype,
            key=keys[1],
        )
        self.a = self._initialize_a(
            (channels,), min_val=a_init_minval, max_val=0.999, dtype=dtype, key=keys[2]
        )

    def _initialize_a(
        self,
        shape: tuple[int],
        min_val: float,
        max_val: float,
        eps=1e-8,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """Initializes the `A` real parameter of the RG-LRU uniformly on a ring, as done
        in the original implementation."""
        unif = jax.random.uniform(
            key, shape=shape, dtype=dtype, minval=min_val**2, maxval=max_val**2
        )
        a = 0.5 * jnp.log(unif + eps)
        return jnp.log(jnp.expm1(-a))

    def __call__(
        self, u: Float[Array, "seq channels"], *, key: PRNGKeyArray | None = None
    ) -> Float[Array, "seq channels"]:
        i_t: Float[Array, "seq channels"] = jax.nn.sigmoid(
            eqx.filter_vmap(self.input_gate)(u)
        )
        r_t: Float[Array, "seq channels"] = jax.nn.sigmoid(
            eqx.filter_vmap(self.recurrence_gate)(u)
        )

        log_a_t: Float[Array, "seq channels"] = -8.0 * r_t * jax.nn.softplus(self.a)
        a_t: Float[Array, "seq channels"] = jnp.exp(log_a_t)
        a_t = a_t.at[0].set(0.0)
        normed_x_t = u * i_t * jnp.sqrt(1 - a_t * a_t)

        y, _ = linear_scan(x=normed_x_t, a=a_t, h0=None, unroll=128)
        return y
