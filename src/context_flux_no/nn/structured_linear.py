from math import sqrt

import equinox as eqx
import jax.numpy as jnp
from einops import rearrange
from equinox._misc import default_floating_dtype
from equinox.nn._misc import default_init, named_scope
from jaxtyping import Array, Float, PRNGKeyArray


class BlockDiagonalLinear(eqx.Module):
    weight: Float[Array, "num_blocks out_features//num_blocks in_feautres//num_blocks"]
    bias: Float[Array, "num_blocks out_features//num_blocks"] | None

    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    num_blocks: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_blocks: int,
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        dtype = default_floating_dtype() if dtype is None else dtype

        if in_features % num_blocks != 0:
            raise ValueError("in_features must be divisable by num_blocks.")
        if out_features % num_blocks != 0:
            raise ValueError("out_features must be divisable by num_blocks.")

        in_dim = in_features // num_blocks
        out_dim = out_features // num_blocks

        wshape = (num_blocks, out_dim, in_dim)
        self.weight = default_init(key, wshape, dtype, 1 / sqrt(in_dim))
        bshape = (num_blocks, out_dim)
        self.bias = jnp.zeros(bshape, dtype=dtype) if use_bias else None

        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        self.use_bias = use_bias

    @named_scope("nn.BlockDiagonalLinear")
    def __call__(
        self, x: Float[Array, " in_features"], *, key: PRNGKeyArray | None = None
    ) -> Float[Array, " out_features"]:
        del key
        x = rearrange(x, "(blocks in_dim) -> blocks in_dim", blocks=self.num_blocks)
        y = jnp.einsum("h i, h o i -> h o", x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return rearrange(y, "blocks out_dim -> (blocks out_dim)")
