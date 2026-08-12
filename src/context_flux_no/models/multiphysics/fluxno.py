from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from ..fno import FNO
from .hyperfluxfno.target_networks import FluxNOTargetNetwork
from .abstract import AbstractMultiphysicsOperator

class NaiveFluxNO(AbstractMultiphysicsOperator):
    fluxno: FluxNOTargetNetwork

    num_spatial_dims: int = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        channels: int,
        lift_dim: int,
        depth: int,
        stencil_widths: tuple[int, int],
        hidden_dim: int,
        stack_grid: bool = True,
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        self.fluxno = FluxNOTargetNetwork(
            num_spatial_dims=num_spatial_dims,
            in_channels=channels,
            out_channels=channels,
            lift_dim=lift_dim,
            depth=depth,
            stencil_widths=stencil_widths,
            hidden_dim=hidden_dim,
            stack_grid=stack_grid,
            activation=activation,
            dtype=dtype,
            key=key
        )
        self.num_spatial_dims = num_spatial_dims

    def __call__(
        self,
        u: Float[Array, "time channels *grids"],
        args: Any = None,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> tuple[Float[Array, " channels *grids"], None]:
        del key, inference

        u0 = u[-1]
        return self.fluxno(u0, args), None