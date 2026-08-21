from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from ..fno import FNO
from .abstract import AbstractMultiphysicsOperator


class NaiveFNO(AbstractMultiphysicsOperator):
    fno: FNO

    num_spatial_dims: int = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        channels: int,
        lift_dim: int,
        depth: int,
        frequency_modes: tuple[int, ...],
        width_lift: int = 128,
        width_project: int = 128,
        depth_lift: int = 1,
        depth_project: int = 1,
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        self.fno = FNO(
            num_spatial_dims=num_spatial_dims,
            in_channels=channels,
            lift_dim=lift_dim,
            depth=depth,
            frequency_modes=frequency_modes,
            fourier_block_type="vanilla",
            width_lift=width_lift,
            width_project=width_project,
            depth_lift=depth_lift,
            depth_project=depth_project,
            activation=activation,
            stack_grid=True,
            residual_connection=True,
            residual_connection_input_output=True,
            dtype=dtype,
            key=key,
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
        del args, key, inference

        u0 = u[-1]
        return self.fno(u0), None


class SpatiotemporalFNO(AbstractMultiphysicsOperator):
    fno: FNO

    num_spatial_dims: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        channels: int,
        lift_dim: int,
        depth: int,
        frequency_modes: tuple[int, ...],
        width_lift: int = 128,
        width_project: int = 128,
        depth_lift: int = 1,
        depth_project: int = 1,
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        self.fno = FNO(
            num_spatial_dims=num_spatial_dims + 1,
            in_channels=channels,
            lift_dim=lift_dim,
            depth=depth,
            frequency_modes=frequency_modes,
            fourier_block_type="vanilla",
            width_lift=width_lift,
            width_project=width_project,
            depth_lift=depth_lift,
            depth_project=depth_project,
            activation=activation,
            stack_grid=True,
            residual_connection=True,
            residual_connection_input_output=True,
            dtype=dtype,
            key=key,
        )
        self.num_spatial_dims = num_spatial_dims
        self.channels = channels

    def __call__(
        self,
        u: Float[Array, "time channels *grids"],
        args: Any = None,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> tuple[Float[Array, " channels *grids"], None]:
        del args, key, inference

        u_out: Float[Array, "channels time *grids"] = self.fno(
            rearrange(u, "t c ... -> c t ...")
        )
        u_out = u_out[:, -1]
        return u_out, None
