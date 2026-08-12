from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray


class ChannelwiseMLP(eqx.Module):
    """A 1x1 convolution-based multilayer perceptron acting on the channel dimension for
    image-like data."""

    layers: tuple[eqx.nn.Conv, ...]

    num_spatial_dims: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    activation: Callable[[Array], Array] = eqx.field(static=True)
    final_activation: Callable[[Array], Array] = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        depth: int,
        activation: Callable = jax.nn.gelu,
        final_activation: Callable = lambda x: x,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        Conv1x1 = partial(
            eqx.nn.Conv, num_spatial_dims=num_spatial_dims, kernel_size=1, dtype=dtype
        )
        keys = jax.random.split(key, depth + 1)

        layers = []
        if depth == 0:
            layers.append(
                Conv1x1(in_channels=in_channels, out_channels=out_channels, key=keys[0])
            )
        else:
            layers.append(
                Conv1x1(
                    in_channels=in_channels, out_channels=hidden_channels, key=keys[0]
                )
            )
            layers.extend(
                Conv1x1(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    key=keys[i],
                )
                for i in range(1, depth)
            )
            layers.append(
                Conv1x1(
                    in_channels=hidden_channels, out_channels=out_channels, key=keys[-1]
                )
            )
        self.layers = tuple(layers)
        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.final_activation = final_activation

    def __call__(
        self,
        x: Float[Array, " in_channels *spatial_dims"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " out_channels *spatial_dims"]:
        del key
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.final_activation(self.layers[-1](x))
