"""Godunov Riemann solvers for scalar conservation laws with a nonconvex flux.

Used by `scripts/data/generate_unseen_flux_1d.py` and
`scripts/data/check_cubic_entropy_fix.py`. The solvers of `cubic.py` and `sine_1d.py`
are left as they were for the ICLR 2027 submission.

For u_t + f(u)_x = 0, the Godunov flux between the states q_l and q_r is

    F*(q_l, q_r) = min of f over [q_l, q_r]   if q_l <= q_r,
                   max of f over [q_r, q_l]   otherwise,

which is the flux of the entropy solution of the Riemann problem for any continuous f
(Osher 1984, SIAM J. Numer. Anal. 21(2); LeVeque 2002, Finite Volume Methods for
Hyperbolic Problems, Section 16.1). The fluctuations

    amdq = F* - f(q_l),    apdq = f(q_r) - F*

sum to f(q_r) - f(q_l), so the wave-propagation update conserves the integral of u.
"""

from collections.abc import Callable

import numpy as np
from clawpack import pyclaw
from jaxtyping import Float


RiemannOutput = tuple[
    Float[np.ndarray, "num_eqns num_waves num_riemanns"],
    Float[np.ndarray, "num_waves num_riemanns"],
    Float[np.ndarray, "num_eqns num_riemanns"],
    Float[np.ndarray, "num_eqns num_riemanns"],
]


def _godunov_fluctuations(
    q_l: Float[np.ndarray, "1 num_riemanns"],
    q_r: Float[np.ndarray, "1 num_riemanns"],
    f_l: Float[np.ndarray, "1 num_riemanns"],
    f_r: Float[np.ndarray, "1 num_riemanns"],
    f_min: Float[np.ndarray, "1 num_riemanns"],
    f_max: Float[np.ndarray, "1 num_riemanns"],
    speed_l: Float[np.ndarray, "1 num_riemanns"],
) -> RiemannOutput:
    """Assemble the PyClaw Riemann output from the range of the flux between the states.

    **Arguments:**

    - `q_l`, `q_r`: the left and right states.
    - `f_l`, `f_r`: the flux at the left and right states.
    - `f_min`, `f_max`: the minimum and maximum of the flux between the two states.
    - `speed_l`: the characteristic speed f'(q_l). It replaces the Rankine-Hugoniot
        speed where the two states coincide.

    **Returns:**

    The tuple `(wave, s, amdq, apdq)`. The wave is the jump `q_r - q_l` and `s` is its
    Rankine-Hugoniot speed, as in `riemann_cubic_1D`. PyClaw uses them for the limited
    second-order correction and for the CFL estimate. The estimate can be below the
    largest characteristic speed between the two states, so the caller keeps the output
    interval short enough to bound the time step.
    """
    dq = q_r - q_l
    is_jump = np.abs(dq) > 1e-14
    s = np.where(is_jump, (f_r - f_l) / np.where(is_jump, dq, 1.0), speed_l)
    f_star = np.where(q_l <= q_r, f_min, f_max)
    wave = np.expand_dims(dq, axis=1)
    return wave, s, f_star - f_l, f_r - f_star


def riemann_sine_godunov_1D(
    q_l: Float[np.ndarray, "1 num_riemanns"],
    q_r: Float[np.ndarray, "1 num_riemanns"],
    aux_l: Float[np.ndarray, "num_aux num_riemanns"],
    aux_r: Float[np.ndarray, "num_aux num_riemanns"],
    problem_data: dict[str, float],
) -> RiemannOutput:
    """Godunov Riemann solver for the sine flux f(u) = a sin(b u).

    The range of the flux between the two states is exact: the sine reaches +1 or -1
    inside an interval of the angle b u exactly when the interval contains a point
    pi/2 + 2 pi k or -pi/2 + 2 pi k.
    """
    a, b = problem_data["a"], problem_data["b"]

    theta_l, theta_r = b * q_l, b * q_r
    theta_lo, theta_hi = np.minimum(theta_l, theta_r), np.maximum(theta_l, theta_r)
    sin_l, sin_r = np.sin(theta_l), np.sin(theta_r)

    half_pi, two_pi = 0.5 * np.pi, 2.0 * np.pi
    has_crest = np.ceil((theta_lo - half_pi) / two_pi) <= np.floor(
        (theta_hi - half_pi) / two_pi
    )
    has_trough = np.ceil((theta_lo + half_pi) / two_pi) <= np.floor(
        (theta_hi + half_pi) / two_pi
    )
    sin_max = np.where(has_crest, 1.0, np.maximum(sin_l, sin_r))
    sin_min = np.where(has_trough, -1.0, np.minimum(sin_l, sin_r))

    return _godunov_fluctuations(
        q_l,
        q_r,
        f_l=a * sin_l,
        f_r=a * sin_r,
        f_min=np.minimum(a * sin_min, a * sin_max),
        f_max=np.maximum(a * sin_min, a * sin_max),
        speed_l=a * b * np.cos(theta_l),
    )


def riemann_cubic_godunov_1D(
    q_l: Float[np.ndarray, "1 num_riemanns"],
    q_r: Float[np.ndarray, "1 num_riemanns"],
    aux_l: Float[np.ndarray, "num_aux num_riemanns"],
    aux_r: Float[np.ndarray, "num_aux num_riemanns"],
    problem_data: dict[str, float],
) -> RiemannOutput:
    """Godunov Riemann solver for the cubic flux f(u) = a u^3 + b u^2 + c u.

    The coefficient names follow `riemann_cubic_1D`. The range of the flux between the
    two states is exact: it is attained at a state or at a root of f' between them.
    """
    a, b, c = problem_data["a"], problem_data["b"], problem_data["c"]

    def flux(u: Float[np.ndarray, "*shape"]) -> Float[np.ndarray, "*shape"]:
        return ((a * u + b) * u + c) * u

    # Real roots of f'(u) = 3 a u^2 + 2 b u + c.
    if a != 0.0:
        discriminant = 4.0 * b * b - 12.0 * a * c
        roots = (
            [(-2.0 * b + sign * np.sqrt(discriminant)) / (6.0 * a) for sign in (-1, 1)]
            if discriminant >= 0.0
            else []
        )
    elif b != 0.0:
        roots = [-c / (2.0 * b)]
    else:
        roots = []

    q_lo, q_hi = np.minimum(q_l, q_r), np.maximum(q_l, q_r)
    f_l, f_r = flux(q_l), flux(q_r)
    # A root clipped to [q_lo, q_hi] is the root itself when it lies between the
    # states and one of the states otherwise, so it never widens the range.
    candidates = [f_l, f_r] + [flux(np.clip(root, q_lo, q_hi)) for root in roots]

    return _godunov_fluctuations(
        q_l,
        q_r,
        f_l=f_l,
        f_r=f_r,
        f_min=np.minimum.reduce(candidates),
        f_max=np.maximum.reduce(candidates),
        speed_l=(3.0 * a * q_l + 2.0 * b) * q_l + c,
    )


def make_scalar_godunov_solver(
    riemann_solver: Callable[..., RiemannOutput],
) -> pyclaw.ClawSolver1D:
    """Return a PyClaw solver for a scalar law with the settings of `CubicFlux1D.solve`:
    the MC limiter, one wave, target CFL number 0.5 and maximum CFL number 0.9."""
    solver = pyclaw.ClawSolver1D(riemann_solver)
    solver.limiters = pyclaw.limiters.tvd.MC
    solver.num_eqn = 1
    solver.num_waves = 1
    solver.kernel_language = "Python"
    solver.cfl_desired = 0.5
    solver.cfl_max = 0.9
    solver.fwave = False
    return solver
