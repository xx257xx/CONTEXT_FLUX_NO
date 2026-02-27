from collections.abc import Callable
from typing import ClassVar, Literal

import equinox as eqx
import numpy as np
import xarray as xr
from clawpack import pyclaw, riemann
from jaxtyping import Float

from ..pdesolve import pdesolve_pyclaw, solution_to_dataset


class Burgers1D(eqx.Module):
    n_dim: ClassVar[int] = 1
    n_eqns: ClassVar[int] = 1
    nu: float = eqx.field(static=True, default=0.0)
    entropy_fix: bool = eqx.field(static=True, default=True)

    @property
    def coeffs(self) -> dict[str, float]:
        return {"nu": self.nu}

    def solve_pyclaw(
        self,
        ic_factory: Callable[[Float[np.ndarray, " Nx"]], Float[np.ndarray, " Nx"]],
        x_span: tuple[float, float],
        Nx: int,
        t_span: tuple[float, float],
        Nt: int,
        bc: Literal["periodic"],
        **pdesolve_kwargs,
    ) -> xr.DataArray:
        solver = pyclaw.ClawSolver1D(riemann.burgers_1D)
        solver.limiters = pyclaw.limiters.tvd.vanleer

        problem_data = {"efix": self.entropy_fix}
        u, t, x_grid = pdesolve_pyclaw(
            solver,
            problem_data,
            ic_factory,
            x_span,
            Nx,
            t_span,
            Nt,
            bc,
            **pdesolve_kwargs,
        )
        return solution_to_dataset(u, t, x_grid, self.coeffs)
