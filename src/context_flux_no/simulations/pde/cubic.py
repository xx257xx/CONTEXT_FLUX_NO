from collections.abc import Callable
from typing import ClassVar, Literal

import equinox as eqx
import numpy as np
from clawpack import pyclaw
from jaxtyping import Float

from ..pdesolve import pdesolve_pyclaw, solution_to_dataset


def riemann_cubic_1D(
    q_l: Float[np.ndarray, "num_eqns num_riemanns"],
    q_r: Float[np.ndarray, "num_eqns num_riemanns"],
    aux_l: Float[np.ndarray, "num_aux num_riemanns"],
    aux_r: Float[np.ndarray, "num_aux num_riemanns"],
    problem_data: dict[str, float],
) -> tuple[
    Float[np.ndarray, "num_eqns num_waves num_riemanns"],
    Float[np.ndarray, "num_waves num_riemanns"],
    Float[np.ndarray, "num_eqns num_riemanns"],
    Float[np.ndarray, "num_eqns num_riemanns"],
]:
    """
    num_riemanns correspond to number of grid cells
    For this problem, num_eqns=1 and num_waves=1.
    """

    a, b, c = problem_data["a"], problem_data["b"], problem_data["c"]

    dq: Float[np.ndarray, "num_eqns num_riemanns"] = q_r - q_l
    # Horner's rule for slightly faster polynomial evaluation
    f_r = ((a * q_r + b) * q_r + c) * q_r
    f_l = ((a * q_l + b) * q_l + c) * q_l
    s_default = (3 * a * q_l + 2 * b) * q_l + c

    s = np.where(np.abs(dq) > 1e-14, (f_r - f_l) / dq, s_default)
    wave = np.expand_dims(dq, axis=1)
    amdq = np.clip(s, max=0.0) * dq
    apdq = np.clip(s, min=0.0) * dq
    return wave, s, amdq, apdq


class CubicFlux1D(eqx.Module):
    n_dim: ClassVar[int] = 1
    n_eqns: ClassVar[int] = 1
    a: float = eqx.field(static=True)
    b: float = eqx.field(static=True)
    c: float = eqx.field(static=True)

    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 1.0):
        self.a = a
        self.b = b
        self.c = c

    @property
    def coeffs(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b, "c": self.c}

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
        solver = pyclaw.ClawSolver1D(riemann_cubic_1D)
        solver.limiters = pyclaw.limiters.tvd.MC
        solver.num_eqn = self.n_eqns
        solver.num_waves = 1
        solver.kernel_language = "Python"
        solver.cfl_desired = 0.5
        solver.cfl_max = 0.9
        solver.fwave = False

        problem_data = {"a": self.a, "b": self.b, "c": self.c}
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
