from collections.abc import Callable
from typing import ClassVar, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from clawpack import pyclaw
from jaxtyping import Array, Float

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


class CubicFlux2D(eqx.Module):
    n_dim: ClassVar[int] = 2
    n_eqns: ClassVar[int] = 1
    a: float = eqx.field(static=True)
    b: float = eqx.field(static=True)
    c: float = eqx.field(static=True)

    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 1.0):
        self.a = a
        self.b = b
        self.c = c

    def flux(self, u: Float[Array, "x y"]) -> Float[Array, "x y"]:
        return ((self.a * u + self.b) * u + self.c) * u

    def flux_speed(self, u: Float[Array, "x y"]) -> Float[Array, "x y"]:
        return (3 * self.a * u + 2 * self.b) * u + self.c

    def _step_rusanov_2d(self, u, dxs, dt):
        # Spatial dimension order is x, y
        dx, dy = dxs

        f = g = self.flux(u)
        s_u_abs = jnp.abs(self.flux_speed(u))

        # x-direction numerical flux at i+1/2 using (u, u_xr)
        u_xr = jnp.roll(u, -1, axis=0)
        f_r = self.flux(u_xr)

        s_xr = jnp.maximum(s_u_abs, jnp.abs(self.flux_speed(u_xr)))
        F_r = 0.5 * (f + f_r) - 0.5 * s_xr * (u_xr - u)

        # x-direction numerical flux at i-1/2 using (u_xl, u)
        u_xl = jnp.roll(u, 1, axis=0)
        f_l = self.flux(u_xl)

        s_xl = jnp.maximum(s_u_abs, jnp.abs(self.flux_speed(u_xl)))
        F_l = 0.5 * (f_l + f) - 0.5 * s_xl * (u - u_xl)

        # y-direction numerical flux at j+1/2 using (u, u_yr)
        u_yr = jnp.roll(u, -1, axis=1)
        g_r = self.flux(u_yr)

        s_yr = jnp.maximum(s_u_abs, jnp.abs(self.flux_speed(u_yr)))
        G_r = 0.5 * (g + g_r) - 0.5 * s_yr * (u_yr - u)

        # y-direction numerical flux at j-1/2 using (u, u_yl)
        u_yl = jnp.roll(u, 1, axis=1)
        g_l = self.flux(u_yl)

        s_yl = jnp.maximum(s_u_abs, jnp.abs(self.flux_speed(u_yl)))
        G_l = 0.5 * (g_l + g) - 0.5 * s_yl * (u - u_yl)

        # Finite volume update
        return u - (dt / dx) * (F_r - F_l) - (dt / dy) * (G_r - G_l)

    def solve(
        self,
        ic_factory: Callable[
            [Float[np.ndarray, " *spatial_dims"]], Float[np.ndarray, " *spatial_dims"]
        ],  # Need to fix function signature here
        x_span: tuple[float, float],
        Nx: int,
        t_span: tuple[float, float],
        Nt: int,
        bc: Literal["periodic"],
        **pdesolve_kwargs,
    ):
        u0 = ic_factory(x, y)

        def _body_fn(u, _):
            u_next = self._step_rusanov_2d(u, (dx, dy), dt)
            return u_next, u_next
