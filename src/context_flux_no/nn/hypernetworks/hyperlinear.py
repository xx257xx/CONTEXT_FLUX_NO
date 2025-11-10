from collections.abc import Callable
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox._misc import default_floating_dtype
from jaxtyping import Array, Float, PRNGKeyArray


class HyperLinear(eqx.Module):
    weight_net: eqx.nn.MLP
    weight_shape: tuple[int, int] = eqx.field(static=True)
    bias_shape: tuple[int] = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hyper_in_dims: int,
        hyper_depth: int,
        hyper_width: int,
        activation: Callable = jax.nn.gelu,
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        dtype = default_floating_dtype() if dtype is None else dtype

        self.weight_shape = (out_features, in_features)
        self.bias_shape = (out_features,)

        out_size = (
            prod(self.weight_shape) + prod(self.bias_shape)
            if use_bias
            else prod(self.weight_shape)
        )
        self.weight_net = eqx.nn.MLP(
            in_size=hyper_in_dims,
            out_size=out_size,
            width_size=hyper_width,
            depth=hyper_depth,
            activation=activation,
            dtype=dtype,
            key=key,
        )

    def __call__(
        self,
        v: Float[Array, " in_features"],
        hyper_input: Float[Array, " hyper_in_dim"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " out_features"]:
        params_flat = self.weight_net(hyper_input)
        weight = jnp.reshape(params_flat[: prod(self.weight_shape)], self.weight_shape)
        v = weight @ v
        if self.use_bias:
            bias = jnp.reshape(params_flat[prod(self.weight_shape) :], self.bias_shape)
            v = v + bias
        return v
