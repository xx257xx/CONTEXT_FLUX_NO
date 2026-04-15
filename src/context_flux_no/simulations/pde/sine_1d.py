from collections.abc import Callable
from typing import ClassVar, Literal

import equinox as eqx
import numpy as np
from clawpack import pyclaw
from jaxtyping import Float

from ..pdesolve import pdesolve_pyclaw, solution_to_dataset


def riemann_sine_1D(q_l, q_r, aux_l, aux_r, problem_data):
    a = problem_data["a"]
    b = problem_data["b"]

    # flux: f(u) = a sin(bu)
    def flux(u):
        return a * np.sin(b * u)

    def dflux(u):
        return a * b * np.cos(b * u)

    # Instead of Roe speed, approximate the characteristic speed at the average position
    u_hat = 0.5 * (q_l[0, :] + q_r[0, :])
    s = dflux(u_hat)[None, :]

    wave = (q_r - q_l)[None, :, :]
    amdq = np.minimum(s, 0.0) * (q_r - q_l)
    apdq = np.maximum(s, 0.0) * (q_r - q_l)

    return wave, s, amdq, apdq


class SineFlux1D(eqx.Module):
    n_dim: ClassVar[int] = 1
    n_eqns: ClassVar[int] = 1
    a: float = eqx.field(static=True)
    b: float = eqx.field(static=True)

    def __init__(self, a: float = 1.0, b: float = 1.0):
        self.a = a
        self.b = b

    @property
    def coeffs(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b}

    def flux(self, u):
        return self.a * np.sin(self.b * u)

    def solve_pyclaw(
        self,
        ic_factory: Callable[[Float[np.ndarray, " Nx"]], Float[np.ndarray, " Nx"]],
        x_span: tuple[float, float],
        Nx: int,
        t_span: tuple[float, float],
        Nt: int,
        bc: Literal["periodic"],
        **pdesolve_kwargs,
    ) -> tuple[
        Float[np.ndarray, "time dim x_grid"],
        Float[np.ndarray, " time"],
        Float[np.ndarray, " x_grid"],
    ]:
        solver = pyclaw.ClawSolver1D(riemann_sine_1D)
        solver.limiters = pyclaw.limiters.tvd.MC
        solver.num_eqn = self.n_eqns
        solver.num_waves = 1
        solver.kernel_language = "Python"
        solver.cfl_desired = 0.5
        solver.cfl_max = 0.9
        solver.fwave = False

        problem_data = {"a": self.a, "b": self.b}
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
