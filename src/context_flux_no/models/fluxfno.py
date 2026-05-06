from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .fno import FNO1D


# TODO: maybe change boundary_conditoin into an enum?
class FluxFNO1D(eqx.Module):
    flux_model: FNO1D

    stencil_size: tuple[int, int] = eqx.field(static=True)
    boundary_condition: Literal["periodic"] = eqx.field(static=True)

    def __init__(
        self,
        data_dim: int,
        lift_dim: int,
        depth: int,
        frequency_modes: int,
        stencil_size: int | tuple[int, int],
        boundary_condition: Literal["periodic"] = "periodic",
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
        self.stencil_size = (
            (stencil_size, stencil_size)
            if isinstance(stencil_size, int)
            else stencil_size
        )
        self.boundary_condition = boundary_condition
        self.flux_model = FNO1D(
            input_dim=data_dim * (self.stencil_size[0] + self.stencil_size[1] + 1),
            lift_dim=lift_dim,
            depth=depth,
            frequency_modes=frequency_modes,
            output_dim=data_dim,
            width_lift=width_lift,
            width_project=width_project,
            depth_lift=depth_lift,
            depth_project=depth_project,
            activation=activation,
            stack_grid=stack_grid,
            dtype=dtype,
            key=key,
        )

    def create_stencil_axis(
        self, v: Float[Array, "data_dim grids"]
    ) -> Float[Array, "data_dim {self.stencil_size[0]}+{self.stencil_size[1]}+2 grids"]:
        assert self.boundary_condition == "periodic", (
            "Other types of boundary conditions are not supported."
        )
        p, q = self.stencil_size
        v_padded: Float[Array, "in_channels grids+p+q+2"] = jnp.concatenate(
            [v[:, -p - 1 :], v, v[:, 0:q]], axis=1
        )
        return jnp.stack(
            [
                jax.lax.dynamic_slice_in_dim(v_padded, i, p + q + 2, axis=-1)
                for i in range(v.shape[1])
            ],
            axis=-1,
        )

    def __call__(
        self, v: Float[Array, "data_dim grids"], dt: float, dx: float
    ) -> Float[Array, "data_dim grids"]:
        v_stencil: Float[
            Array, "data_dim {self.stencil_size[0]}+{self.stencil_size[1]}+2 grids"
        ] = self.create_stencil_axis(v)
        n_grids = v.shape[1]
        f_l = self.flux_model(jnp.reshape(v_stencil[:, :-1], (-1, n_grids)))
        f_r = self.flux_model(jnp.reshape(v_stencil[:, 1:], (-1, n_grids)))
        return v + dt * (f_l - f_r) / dx

    def physical_flux(
        self, v: Float[Array, "data_dim grids"]
    ) -> Float[Array, "data_dim grids"]:
        n_repeat = self.stencil_size[0] + self.stencil_size[1] + 1
        return self.flux_model(jnp.repeat(v, n_repeat, axis=0))
