import abc
from collections.abc import Callable
from typing import Literal, TypedDict

import equinox as eqx
import numpy as np
from einops import rearrange
from jaxtyping import Array, Float


class WellSchema(TypedDict):
    scalars: dict[str, np.ndarray]
    t0_fields: dict[str, np.ndarray]
    t1_fields: dict[str, np.ndarray]
    t2_fields: dict[str, np.ndarray]


class AbstractHyperbolicConservationLaw(eqx.Module):
    """Abstract base class for hyperbolic conservation laws, to serve as an interface
    for downstream functionalities regarding dataset generation."""

    n_spatial_dims: eqx.AbstractVar[int]
    parameters: eqx.AbstractVar[dict[str, float]]
    field_rank_names: tuple[tuple[int, str], ...]

    @property
    def n_eqns(self) -> int:
        return sum(self.n_spatial_dims**r for r, _ in self.field_rank_names)

    @abc.abstractmethod
    def solve(
        self,
        ic_factory: Callable[[Float[np.ndarray, " Nx"]], Float[np.ndarray, " Nx"]],
        x_span: tuple[float, float],
        Nx: int,
        t_span: tuple[float, float],
        Nt: int,
        bc: Literal["periodic"],  # TODO: extend to other types as well
        **pdesolve_kwargs,
    ):
        # TODO: generalize annotations for x_span, Nx for general N-D case
        # TODO: annotate return type
        pass

    def solution_to_well_schema(self, u: Float[Array, "time dim *x_grids"]):
        well_schema = WellSchema(scalars={}, t0_fields={}, t1_fields={}, t2_fields={})

        idx_start = 0
        for rank, f_name in self.field_rank_names:
            idx_end = idx_start + self.n_spatial_dims**rank
            field_data = rearrange(u[:, idx_start:idx_end], "T C ... -> T ... C")
            if rank == 0:
                field_data = np.squeeze(field_data, axis=-1)
            well_schema[f"t{rank}_fields"][f_name] = field_data

        for param_name, val in self.parameters.items():
            well_schema["scalars"][param_name] = val
        return well_schema
