"""Compressible ideal-gas Euler equations with one physical coefficient.

``gamma``
    Ratio of specific heats.  ``gamma=1.4`` is the usual ideal-air value.

The equation of state is

    p = (gamma - 1) * (E - kinetic_energy).

Consequently, primitive initial data must be converted with

    E = p / (gamma - 1) + kinetic_energy.

The 1-D solver uses an HLL Riemann solver through PyClaw.  The 2-D solver
uses a dimension-by-dimension local Lax--Friedrichs (Rusanov) flux and
periodic boundaries, matching the periodic ``jnp.roll`` convention used by
the original scalar-flux implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from clawpack import pyclaw
from jaxtyping import Array, Float

from ..pdesolve import pdesolve_pyclaw, solution_to_dataset


_RHO_FLOOR = 1.0e-12
_PRESSURE_FLOOR = 1.0e-12
_SPEED_GAP_FLOOR = 1.0e-12


def _pressure_flux_1d(
    q: Float[np.ndarray, "3 num_riemanns"],
    gamma: float,
) -> tuple[
    Float[np.ndarray, "num_riemanns"],
    Float[np.ndarray, "3 num_riemanns"],
    Float[np.ndarray, "num_riemanns"],
]:
    """Return pressure, physical flux, and sound speed for 1-D states."""

    rho = np.maximum(q[0], _RHO_FLOOR)
    momentum = q[1]
    energy = q[2]
    velocity = momentum / rho
    kinetic = 0.5 * momentum * velocity
    pressure = (gamma - 1.0) * (energy - kinetic)

    flux = np.stack(
        (
            momentum,
            momentum * velocity + pressure,
            velocity * (energy + pressure),
        ),
        axis=0,
    )
    sound_speed = np.sqrt(gamma * np.maximum(pressure, _PRESSURE_FLOOR) / rho)
    return pressure, flux, sound_speed


def riemann_euler_hll_1D(
    q_l: Float[np.ndarray, "3 num_riemanns"],
    q_r: Float[np.ndarray, "3 num_riemanns"],
    aux_l: Float[np.ndarray, "num_aux num_riemanns"],
    aux_r: Float[np.ndarray, "num_aux num_riemanns"],
    problem_data: dict[str, float],
) -> tuple[
    Float[np.ndarray, "3 2 num_riemanns"],
    Float[np.ndarray, "2 num_riemanns"],
    Float[np.ndarray, "3 num_riemanns"],
    Float[np.ndarray, "3 num_riemanns"],
]:
    """Two-wave HLL Riemann solver for the 1-D Euler equations.

    PyClaw passes left and right states with shape
    ``(num_eqns, num_riemanns)``.  ``aux_l`` and ``aux_r`` are intentionally
    unused because ``gamma`` is spatially constant.
    """

    del aux_l, aux_r
    gamma = problem_data["gamma"]

    _, flux_l, sound_l = _pressure_flux_1d(q_l, gamma)
    _, flux_r, sound_r = _pressure_flux_1d(q_r, gamma)
    velocity_l = q_l[1] / np.maximum(q_l[0], _RHO_FLOOR)
    velocity_r = q_r[1] / np.maximum(q_r[0], _RHO_FLOOR)

    speed_l = np.minimum(velocity_l - sound_l, velocity_r - sound_r)
    speed_r = np.maximum(velocity_l + sound_l, velocity_r + sound_r)
    speed_gap = np.maximum(speed_r - speed_l, _SPEED_GAP_FLOOR)

    dq = q_r - q_l
    dflux = flux_r - flux_l
    wave_l = (speed_r[None, :] * dq - dflux) / speed_gap[None, :]
    wave_r = (dflux - speed_l[None, :] * dq) / speed_gap[None, :]
    wave = np.stack((wave_l, wave_r), axis=1)
    speeds = np.stack((speed_l, speed_r), axis=0)

    amdq = np.sum(np.minimum(speeds, 0.0)[None, :, :] * wave, axis=1)
    apdq = np.sum(np.maximum(speeds, 0.0)[None, :, :] * wave, axis=1)
    return wave, speeds, amdq, apdq


class Euler1D(eqx.Module):
    """One-dimensional compressible Euler equation.

    Conservative state order: ``(rho, rho*u, E)``.
    """

    n_dim: ClassVar[int] = 1
    n_eqns: ClassVar[int] = 3
    gamma: float = eqx.field(static=True)

    def __init__(self, gamma: float = 1.4):
        if gamma <= 1.0:
            raise ValueError("gamma must be greater than 1")
        self.gamma = gamma

    @property
    def coeffs(self) -> dict[str, float]:
        return {"gamma": self.gamma}

    def primitive_to_conservative(
        self,
        rho: Float[Array, "..."],
        velocity: Float[Array, "..."],
        pressure: Float[Array, "..."],
    ) -> Float[Array, "3 ..."]:
        """Convert ``(rho, u, p)`` to ``(rho, rho*u, E)``."""

        rho = jnp.asarray(rho)
        velocity = jnp.asarray(velocity)
        pressure = jnp.asarray(pressure)
        momentum = rho * velocity
        energy = pressure / (self.gamma - 1.0) + 0.5 * rho * velocity**2
        return jnp.stack((rho, momentum, energy), axis=0)

    def solve(
        self,
        ic_factory: Callable[[Float[np.ndarray, "Nx"]], Float[np.ndarray, "3 Nx"]],
        x_span: tuple[float, float],
        Nx: int,
        t_span: tuple[float, float],
        Nt: int,
        bc: Literal["periodic"],
        **pdesolve_kwargs,
    ):
        solver = pyclaw.ClawSolver1D(riemann_euler_hll_1D)
        solver.limiters = pyclaw.limiters.tvd.MC
        solver.num_eqn = self.n_eqns
        solver.num_waves = 2
        solver.kernel_language = "Python"
        solver.cfl_desired = 0.45
        solver.cfl_max = 0.9
        solver.fwave = False

        problem_data = {"gamma": self.gamma}
        q, t, x_grid = pdesolve_pyclaw(
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
        return solution_to_dataset(q, t, (x_grid,), self.coeffs)


class Euler2D(eqx.Module):
    """Two-dimensional compressible Euler equation with periodic boundaries.

    Conservative state order: ``(rho, rho*u, rho*v, E)``.  Arrays use the
    layout ``(equation, x, y)``.
    """

    n_dim: ClassVar[int] = 2
    n_eqns: ClassVar[int] = 4
    gamma: float = eqx.field(static=True)

    def __init__(self, gamma: float = 1.4):
        if gamma <= 1.0:
            raise ValueError("gamma must be greater than 1")
        self.gamma = gamma

    @property
    def coeffs(self) -> dict[str, float]:
        return {"gamma": self.gamma}

    def primitive_to_conservative(
        self,
        rho: Float[Array, "..."],
        velocity_x: Float[Array, "..."],
        velocity_y: Float[Array, "..."],
        pressure: Float[Array, "..."],
    ) -> Float[Array, "4 ..."]:
        """Convert ``(rho, u, v, p)`` to ``(rho, rho*u, rho*v, E)``."""

        rho = jnp.asarray(rho)
        velocity_x = jnp.asarray(velocity_x)
        velocity_y = jnp.asarray(velocity_y)
        pressure = jnp.asarray(pressure)
        momentum_x = rho * velocity_x
        momentum_y = rho * velocity_y
        energy = pressure / (self.gamma - 1.0) + 0.5 * rho * (
            velocity_x**2 + velocity_y**2
        )
        return jnp.stack((rho, momentum_x, momentum_y, energy), axis=0)

    def pressure(self, q: Float[Array, "4 x y"]) -> Float[Array, "x y"]:
        rho = jnp.maximum(q[0], _RHO_FLOOR)
        kinetic = 0.5 * (q[1] ** 2 + q[2] ** 2) / rho
        return (self.gamma - 1.0) * (q[3] - kinetic)

    def flux_x(self, q: Float[Array, "4 x y"]) -> Float[Array, "4 x y"]:
        rho = jnp.maximum(q[0], _RHO_FLOOR)
        velocity_x = q[1] / rho
        pressure = self.pressure(q)
        return jnp.stack(
            (
                q[1],
                q[1] * velocity_x + pressure,
                q[2] * velocity_x,
                velocity_x * (q[3] + pressure),
            ),
            axis=0,
        )

    def flux_y(self, q: Float[Array, "4 x y"]) -> Float[Array, "4 x y"]:
        rho = jnp.maximum(q[0], _RHO_FLOOR)
        velocity_y = q[2] / rho
        pressure = self.pressure(q)
        return jnp.stack(
            (
                q[2],
                q[1] * velocity_y,
                q[2] * velocity_y + pressure,
                velocity_y * (q[3] + pressure),
            ),
            axis=0,
        )

    def signal_speeds(
        self, q: Float[Array, "4 x y"]
    ) -> tuple[Float[Array, "x y"], Float[Array, "x y"]]:
        rho = jnp.maximum(q[0], _RHO_FLOOR)
        velocity_x = q[1] / rho
        velocity_y = q[2] / rho
        pressure = jnp.maximum(self.pressure(q), _PRESSURE_FLOOR)
        sound_speed = jnp.sqrt(self.gamma * pressure / rho)
        return jnp.abs(velocity_x) + sound_speed, jnp.abs(velocity_y) + sound_speed

    def _rusanov_flux_x(
        self,
        q_l: Float[Array, "4 x y"],
        q_r: Float[Array, "4 x y"],
    ) -> Float[Array, "4 x y"]:
        speed_l, _ = self.signal_speeds(q_l)
        speed_r, _ = self.signal_speeds(q_r)
        speed = jnp.maximum(speed_l, speed_r)
        return 0.5 * (self.flux_x(q_l) + self.flux_x(q_r)) - 0.5 * speed * (
            q_r - q_l
        )

    def _rusanov_flux_y(
        self,
        q_l: Float[Array, "4 x y"],
        q_r: Float[Array, "4 x y"],
    ) -> Float[Array, "4 x y"]:
        _, speed_l = self.signal_speeds(q_l)
        _, speed_r = self.signal_speeds(q_r)
        speed = jnp.maximum(speed_l, speed_r)
        return 0.5 * (self.flux_y(q_l) + self.flux_y(q_r)) - 0.5 * speed * (
            q_r - q_l
        )

    def _step_rusanov_2d(
        self,
        q: Float[Array, "4 x y"],
        dxs: tuple[float, float],
        dt: float,
    ) -> Float[Array, "4 x y"]:
        dx, dy = dxs

        # Periodic neighbors.  Spatial axes are 1 and 2 because axis 0 stores
        # the four conservative variables.
        q_xr = jnp.roll(q, -1, axis=1)
        q_xl = jnp.roll(q, 1, axis=1)
        flux_x_r = self._rusanov_flux_x(q, q_xr)
        flux_x_l = self._rusanov_flux_x(q_xl, q)

        q_yr = jnp.roll(q, -1, axis=2)
        q_yl = jnp.roll(q, 1, axis=2)
        flux_y_r = self._rusanov_flux_y(q, q_yr)
        flux_y_l = self._rusanov_flux_y(q_yl, q)

        return q - (dt / dx) * (flux_x_r - flux_x_l) - (dt / dy) * (
            flux_y_r - flux_y_l
        )

    def _validate_state(self, q: Array, *, where: str) -> None:
        if q.ndim != 3 or q.shape[0] != self.n_eqns:
            raise ValueError(
                f"{where} must have shape (4, Nx, Ny); received {tuple(q.shape)}"
            )
        min_rho = float(jnp.min(q[0]))
        min_pressure = float(jnp.min(self.pressure(q)))
        if min_rho <= 0.0:
            raise ValueError(f"{where} has non-positive density ({min_rho})")
        if min_pressure <= 0.0:
            raise ValueError(f"{where} has non-positive pressure ({min_pressure})")

    def solve(
        self,
        ic_factory: Callable[
            [tuple[Float[Array, "Nx"], Float[Array, "Ny"]]],
            Float[Array, "4 Nx Ny"],
        ],
        x_spans: tuple[tuple[float, float], tuple[float, float]],
        Nxs: tuple[int, int],
        t_span: tuple[float, float],
        Nt: int,
        *,
        cfl: float = 0.4,
        max_substeps_per_output: int = 100_000,
        **kwargs,
    ):
        """Integrate with adaptive CFL substeps and return ``Nt+1`` snapshots.

        ``kwargs`` is accepted for interface compatibility with the original
        class.  The finite-volume boundary condition is periodic.
        """

        del kwargs
        if not (0.0 < cfl <= 1.0):
            raise ValueError("cfl must lie in (0, 1]")
        if Nt <= 0 or min(Nxs) <= 0:
            raise ValueError("Nt, Nx, and Ny must be positive")

        Nx, Ny = Nxs
        x_span, y_span = x_spans
        dx = (x_span[1] - x_span[0]) / Nx
        dy = (y_span[1] - y_span[0]) / Ny
        if dx <= 0.0 or dy <= 0.0:
            raise ValueError("each spatial span must be strictly increasing")

        t0, t1 = t_span
        if t1 <= t0:
            raise ValueError("t_span must be strictly increasing")

        x = jnp.linspace(x_span[0] + 0.5 * dx, x_span[1] - 0.5 * dx, Nx)
        y = jnp.linspace(y_span[0] + 0.5 * dy, y_span[1] - 0.5 * dy, Ny)
        t = jnp.linspace(t0, t1, Nt + 1)

        q = jnp.asarray(ic_factory((x, y)))
        self._validate_state(q, where="initial state")
        snapshots = [q]

        output_dt = (t1 - t0) / Nt
        time_tolerance = 16.0 * np.finfo(float).eps * max(1.0, abs(output_dt))
        for output_index in range(Nt):
            remaining = output_dt
            substeps = 0
            while remaining > time_tolerance:
                speed_x, speed_y = self.signal_speeds(q)
                inverse_dt = float(jnp.max(speed_x)) / dx + float(
                    jnp.max(speed_y)
                ) / dy
                dt_cfl = cfl / inverse_dt if inverse_dt > 0.0 else remaining
                dt = min(remaining, dt_cfl)
                if not np.isfinite(dt) or dt <= 0.0:
                    raise RuntimeError("CFL calculation produced an invalid time step")

                q = self._step_rusanov_2d(q, (dx, dy), dt)
                remaining -= dt
                substeps += 1
                if substeps > max_substeps_per_output:
                    raise RuntimeError(
                        "maximum CFL substeps exceeded before output "
                        f"{output_index + 1}"
                    )

            self._validate_state(q, where=f"state at output {output_index + 1}")
            snapshots.append(q)

        trajectory: Float[Array, "Nt_plus_1 4 Nx Ny"] = jnp.stack(
            snapshots, axis=0
        )
        return solution_to_dataset(trajectory, t, (x, y), self.coeffs)
