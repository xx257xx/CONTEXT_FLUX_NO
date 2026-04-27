from collections.abc import Callable
from typing import ClassVar, Literal

import equinox as eqx
import numpy as np
from clawpack import pyclaw
from jaxtyping import Float

from ..pdesolve import pdesolve_pyclaw, solution_to_dataset


def riemann_param_shallow_water_1D(
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
    Roe-type approximate Riemann solver for a 3-parameter
    shallow-water-like 1D conservation law.

    State:
        q[0] = h
        q[1] = m = h u

    Flux:
        f(q) = [
            gamma * m,
            alpha * m^2 / h + 0.5 * beta * h^2
        ]

    Standard shallow water:
        gamma = 1
        alpha = 1
        beta = g

    Jacobian:
        A(q) = [
            [0, gamma],
            [beta h - alpha u^2, 2 alpha u]
        ]

    Eigenvalues:
        lambda = alpha u ± sqrt(alpha^2 u^2
                                - gamma (alpha u^2 - beta h))

               = alpha u ± sqrt(alpha (alpha - gamma) u^2
                                + gamma beta h)

    Hyperbolicity is safest when:
        gamma > 0, beta > 0, alpha close to gamma.
    """

    gamma = problem_data.get("gamma", 1.0)
    alpha = problem_data.get("alpha", 1.0)
    beta = problem_data.get("beta", 1.0)
    h_floor = problem_data.get("h_floor", 1e-8)

    h_l = np.maximum(q_l[0], h_floor)
    h_r = np.maximum(q_r[0], h_floor)

    m_l = q_l[1]
    m_r = q_r[1]

    u_l = m_l / h_l
    u_r = m_r / h_r

    dh = h_r - h_l
    dm = m_r - m_l

    sqrt_h_l = np.sqrt(h_l)
    sqrt_h_r = np.sqrt(h_r)

    u_hat = (sqrt_h_l * u_l + sqrt_h_r * u_r) / (sqrt_h_l + sqrt_h_r)
    h_hat = 0.5 * (h_l + h_r)

    c2_hat = alpha * (alpha - gamma) * u_hat**2 + gamma * beta * h_hat
    c_hat = np.sqrt(np.maximum(c2_hat, h_floor))

    s1 = alpha * u_hat - c_hat
    s2 = alpha * u_hat + c_hat

    denom = gamma * (s2 - s1)
    denom = np.where(np.abs(denom) > 1e-14, denom, 1e-14)

    # Eigenvectors can be chosen as:
    # r_p = [gamma, s_p]^T
    #
    # dq = a1 r1 + a2 r2
    alpha2_wave = (gamma * dm - s1 * dh) / denom
    alpha1_wave = (dh / gamma) - alpha2_wave

    num_riemanns = q_l.shape[1]

    wave = np.empty((2, 2, num_riemanns), dtype=q_l.dtype)

    wave[0, 0, :] = alpha1_wave * gamma
    wave[1, 0, :] = alpha1_wave * s1

    wave[0, 1, :] = alpha2_wave * gamma
    wave[1, 1, :] = alpha2_wave * s2

    s = np.stack([s1, s2], axis=0)

    amdq = (
        np.minimum(s1, 0.0)[None, :] * wave[:, 0, :]
        + np.minimum(s2, 0.0)[None, :] * wave[:, 1, :]
    )

    apdq = (
        np.maximum(s1, 0.0)[None, :] * wave[:, 0, :]
        + np.maximum(s2, 0.0)[None, :] * wave[:, 1, :]
    )

    return wave, s, amdq, apdq


class ParametrizedShallowWater1D(eqx.Module):
    n_dim: ClassVar[int] = 1
    n_eqns: ClassVar[int] = 2

    gamma: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    h_floor: float = eqx.field(static=True)

    def __init__(
        self,
        gamma: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        h_floor: float = 1e-8,
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.h_floor = h_floor

    @property
    def coeffs(self) -> dict[str, float]:
        return {
            "gamma": self.gamma,
            "alpha": self.alpha,
            "beta": self.beta,
            "h_floor": self.h_floor,
        }

    def solve(
        self,
        ic_factory: Callable[
            [Float[np.ndarray, " Nx"]],
            Float[np.ndarray, " dim Nx"],
        ],
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
        solver = pyclaw.ClawSolver1D(riemann_param_shallow_water_1D)

        solver.limiters = pyclaw.limiters.tvd.MC
        solver.num_eqn = self.n_eqns
        solver.num_waves = 2
        solver.kernel_language = "Python"
        solver.cfl_desired = 0.5
        solver.cfl_max = 0.9
        solver.fwave = False

        problem_data = {
            "gamma": self.gamma,
            "alpha": self.alpha,
            "beta": self.beta,
            "h_floor": self.h_floor,
        }

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

        return solution_to_dataset(u, t, (x_grid,), self.coeffs)
