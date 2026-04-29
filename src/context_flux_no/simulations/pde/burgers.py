from collections.abc import Callable
from typing import ClassVar, Literal

import equinox as eqx
import jax.numpy as jnp
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

    def solve(
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


class ViscousBurgers1D(eqx.Module):
    """
    Parametric viscous Burgers-type equation:

        u_t + a (u^2)_x = b u_xx

    Equivalently:

        u_t + f(u)_x = b u_xx,
        f(u) = a u^2.

    Parameters:
        a: nonlinear advection coefficient
        b: diffusion/viscosity coefficient

    This solver assumes periodic boundary conditions.
    """

    n_dim: ClassVar[int] = 1
    n_eqns: ClassVar[int] = 1

    a: float = eqx.field(static=True)
    b: float = eqx.field(static=True)

    def __init__(self, a: float = 1.0, b: float = 0.01):
        self.a = a
        self.b = b

    @property
    def coeffs(self) -> dict[str, float]:
        return {
            "a": self.a,
            "b": self.b,
        }

    def flux(
        self,
        u: Float[np.ndarray, "Nx"],
    ) -> Float[np.ndarray, "Nx"]:
        return self.a * u**2

    def solve(
        self,
        ic_factory: Callable[[Float[np.ndarray, " Nx"]], Float[np.ndarray, " Nx"]],
        x_span: tuple[float, float],
        Nx: int,
        t_span: tuple[float, float],
        Nt: int,
        bc: Literal["periodic"],
        cfl: float = 0.4,
        max_steps: int = 10_000_000,
        **kwargs,
    ) -> tuple[
        Float[np.ndarray, "time dim x_grid"],
        Float[np.ndarray, " time"],
        Float[np.ndarray, " x_grid"],
    ]:
        """
        Solves

            u_t + a (u^2)_x = b u_xx

        using:
            - finite volume Rusanov flux for the nonlinear hyperbolic term
            - centered finite difference for diffusion
            - explicit time stepping
            - periodic boundary condition

        Returns:
            solution_to_dataset(u, t, x_grid, self.coeffs)

        Expected raw solution shape before solution_to_dataset:
            u_save.shape = (Nt, 1, Nx)
        """

        if bc != "periodic":
            raise NotImplementedError(
                "Currently only periodic boundary conditions are supported."
            )

        x0, x1 = x_span
        t0, t1 = t_span

        dx = (x1 - x0) / Nx
        x_grid = np.linspace(x0, x1, Nx, endpoint=False)

        t_save = np.linspace(t0, t1, Nt + 1)

        u = jnp.squeeze(ic_factory(x_grid))

        if u.ndim != 1:
            raise ValueError(
                f"ic_factory must return shape (Nx,), but got shape {u.shape}."
            )

        u_save = np.zeros((Nt + 1, 1, Nx), dtype=float)
        u_save[0, 0, :] = u

        t = t0
        save_idx = 1
        step = 0

        while save_idx < Nt:
            if step > max_steps:
                raise RuntimeError(
                    f"Maximum number of steps exceeded: {max_steps}. "
                    "Try increasing b, decreasing Nx, or increasing cfl carefully."
                )

            # Characteristic speed for f(u) = a u^2:
            # f'(u) = 2 a u
            max_adv_speed = np.max(np.abs(2.0 * self.a * u))

            if max_adv_speed > 1e-14:
                dt_adv = dx / max_adv_speed
            else:
                dt_adv = np.inf

            if self.b > 1e-14:
                dt_diff = 0.5 * dx**2 / self.b
            else:
                dt_diff = np.inf

            dt = cfl * min(dt_adv, dt_diff)

            next_save_t = t_save[save_idx]
            if t + dt > next_save_t:
                dt = next_save_t - t

            if dt <= 0:
                u_save[save_idx, 0, :] = u
                save_idx += 1
                continue

            u = self._step_periodic(u, dx, dt)

            t += dt
            step += 1

            if abs(t - next_save_t) < 1e-12:
                u_save[save_idx, 0, :] = u
                save_idx += 1

        return solution_to_dataset(u_save, t_save, (x_grid,), self.coeffs)

    def _step_periodic(
        self,
        u: Float[np.ndarray, " Nx"],
        dx: float,
        dt: float,
    ) -> Float[np.ndarray, " Nx"]:
        """
        One explicit finite-volume / finite-difference step.

        Hyperbolic part:
            u_t + f(u)_x = 0

        Diffusion part:
            u_t = b u_xx
        """

        u_l = u
        u_r = np.roll(u, -1)

        f_l = self.flux(u_l)
        f_r = self.flux(u_r)

        # Local Rusanov speed at each interface i+1/2.
        # max |f'(u)| = max |2 a u|
        smax = np.maximum(
            np.abs(2.0 * self.a * u_l),
            np.abs(2.0 * self.a * u_r),
        )

        # Numerical flux F_{i+1/2}
        flux_half = 0.5 * (f_l + f_r) - 0.5 * smax * (u_r - u_l)

        # Conservative hyperbolic update:
        # -(F_{i+1/2} - F_{i-1/2}) / dx
        hyperbolic_rhs = -(flux_half - np.roll(flux_half, 1)) / dx

        # Centered diffusion:
        # b * (u_{i+1} - 2u_i + u_{i-1}) / dx^2
        diffusion_rhs = self.b * (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / dx**2

        return u + dt * (hyperbolic_rhs + diffusion_rhs)
