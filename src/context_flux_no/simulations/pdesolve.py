from collections.abc import Callable
from typing import Literal

import numpy as np
from clawpack import pyclaw
from jaxtyping import Float

from .pyclaw_utils import (
    apply_initial_condition,
    bc_from_string,
    grid_centers_from_state,
    make_controller,
    make_domain,
    solution_from_controller,
)


ClawSolver = pyclaw.ClawSolver1D | pyclaw.ClawSolver2D | pyclaw.ClawSolver3D


def pdesolve_pyclaw(
    solver: ClawSolver,
    problem_data: dict[str, float],
    ic_factory: Callable[[Float[np.ndarray, " Nx"]], Float[np.ndarray, " Nx"]],
    x_span: tuple[float, float],
    Nx: int,
    t_span: tuple[float, float],
    Nt: int,
    bc: Literal["periodic"],
) -> tuple[
    Float[np.ndarray, "time dim x_grid"],
    Float[np.ndarray, " time"],
    Float[np.ndarray, " x_grid"],
]:
    # Need to change for >1D cases
    solver.bc_lower[0] = solver.bc_upper[0] = bc_from_string(bc)
    domain = make_domain(x_span, Nx)

    state = pyclaw.State(domain, solver.num_eqn)
    state.problem_data.update(problem_data)
    apply_initial_condition(state, ic_factory)

    controller = make_controller(state, domain, t_span, Nt)
    controller.solver = solver

    _ = controller.run()

    u = solution_from_controller(controller)
    t = np.asarray(controller.out_times)
    (x_grid,) = grid_centers_from_state(state)
    return u, t, x_grid
