from typing import Any

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

from .abstract import AbstractMultiphysicsOperator


class MultiphysicsWrapper(AbstractMultiphysicsOperator):
    """A wrapper class that wraps a neural operator model that maps a solution u_t to
    u_{t_\Delta t} to a multiphysics model.

    Note that this is achieved by disregarding the context, and just mapping the last
    time step of the given context trajectory.
    As such, models created with this wrapper are not expect to work properly.
    Rather, this class is intended to be used for ablation experiment purposes."""

    neural_operator: eqx.Module
    num_spatial_dims: int = eqx.field(static=True)

    def __init__(
        self, neural_operator: eqx.Module, num_spatial_dims: int | None = None
    ):
        self.neural_operator = neural_operator
        if hasattr(neural_operator, "num_spatial_dims"):
            self.num_spatial_dims = neural_operator.num_spatial_dims
        else:
            assert (
                num_spatial_dims is not None
            ), """num_spatial_dims must be explicitly provided for neural operator 
                models not having it as a property."""
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
        return self.neural_operator(u0), None
