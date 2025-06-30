from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..nn import Fourier1D


## TODO: Implement FNO and FluxFNO, test on Burgers to debug and test performance
## TODO: Add padding?
class FNO1D(eqx.Module, strict=True):
    lift_layer: eqx.nn.MLP
    fourier_layers = tuple[Fourier1D, ...]
    project_layer: eqx.nn.MLP
    activation: Callable
    data_dim: int = eqx.field(static=True)
    lift_dim: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    frequency_modes: int = eqx.field(static=True)
    width_lift: int = eqx.field(static=True)
    width_project: int = eqx.field(static=True)
    depth_lift: int = eqx.field(static=True)
    depth_project: int = eqx.field(static=True)
    stack_grid: bool = eqx.field(static=True)

    def __init__(
        self,
        data_dim: int,
        lift_dim: int,
        depth: int,
        frequency_modes: int,
        width_lift: int = 128,
        width_project: int = 128,
        depth_lift: int = 1,
        depth_project: int = 1,
        activation: Callable = jax.nn.gelu,
        stack_grid: bool = True,
        dtype=None,
        *,
        key,
    ):
        keys = jax.random.split(key, depth + 2)

        self.lift_layer = eqx.nn.MLP(
            data_dim + 1 if stack_grid else data_dim,
            lift_dim,
            width_lift,
            depth_lift,
            activation,
            dtype=dtype,
            key=keys[0],
        )
        self.fourier_layers = tuple(
            Fourier1D(
                lift_dim, lift_dim, frequency_modes, activation, dtype=dtype, key=k
            )
            for k in keys[1:-1]
        )
        self.project_layer = eqx.nn.MLP(
            lift_dim,
            data_dim,
            width_project,
            depth_project,
            activation,
            dtype=dtype,
            key=keys[-1],
        )

        self.data_dim = data_dim
        self.lift_dim = lift_dim
        self.depth = depth
        self.frequency_modes = frequency_modes
        self.width_lift = width_lift
        self.width_project = width_project
        self.depth_lift = depth_lift
        self.depth_project = depth_project

    @property
    def layers(self) -> tuple[eqx.Module, ...]:
        return (self.lift_layer, *self.fourier_layers, self.project_layer)

    def forward(
        self, v: Float[Array, "in_channels grids"]
    ) -> Float[Array, "out_channels grids"]:
        if self.stack_grid:
            grid = jnp.linspace(0, 1, v.shape[-1])
            v = jnp.stack((v, grid), axis=0)

        v = eqx.filter_vmap(self.lift_layer, axis=-1)(v)
        for fourier in self.fourier_layers:
            v = fourier(v)
        v = eqx.filter_vmap(self.project_layer, axis=-1)(v)
        return v
