from collections.abc import Callable

import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

from context_flux_no.models.fno import FNO

from .base import AbstractTargetNetwork


class FNOTargetNetwork(AbstractTargetNetwork):
    _fno: FNO

    num_spatial_dims: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
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
        self._fno = FNO(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
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
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __call__(
        self,
        u: Float[Array, " in_channels *spatial_dims"],
        args: tuple[float, ...],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Float[Array, " out_channels *spatial_dims"]:
        del args, key, inference
        return self._fno(u)
