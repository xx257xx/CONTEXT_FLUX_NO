from collections.abc import Callable

import numpy as np
from clawpack import pyclaw
from jaxtyping import Float


def bc_from_string(bc_name: str) -> pyclaw.BC:
    match bc_name:
        case "periodic":
            bc_pyclaw = pyclaw.BC.periodic
        case _:
            raise ValueError("Unrecognized boundary condition")
    return bc_pyclaw


def make_controller(
    state: pyclaw.State, domain: pyclaw.Domain, t_span: tuple[float, float], Nt: int
) -> pyclaw.Controller:
    controller = pyclaw.Controller()
    controller.solution = pyclaw.Solution(state, domain)
    controller.tfinal = t_span[1] - t_span[0]
    controller.num_output_times = Nt
    # By default, keep the solution in memory and do not save
    # See https://www.clawpack.org/pyclaw/output.html for details
    controller.keep_copy = True
    controller.output_format = None
    return controller


def solution_from_controller(
    controller: pyclaw.Controller,
) -> Float[np.ndarray, "time dim *grids"]:
    return np.stack([f.state.q for f in controller.frames], axis=0)


def grid_centers_from_state(
    state: pyclaw.State,
) -> list[Float[np.ndarray, " grid"]]:
    grid = state.grid
    grid_centers = [getattr(grid, dim_name).centers for dim_name in grid._dimensions]
    return grid_centers


def make_domain(x_span, Nx):
    match x_span, Nx:
        case tuple([float() | int(), float() | int()]), int():
            dim = pyclaw.Dimension(*x_span, Nx, name="x")
        case [[tuple([float() | int(), float() | int()]), *_], int()]:
            dim = [
                pyclaw.Dimension(*x_span_i, Nx, name=f"x_{i}")
                for i, x_span_i in enumerate(x_span)
            ]
        case [[tuple([float() | int(), float() | int()]), *_], [int(), _]]:
            dim = [
                pyclaw.Dimension(*x_span_i, Nx_i, name=f"x_{i}")
                for i, (x_span_i, Nx_i) in enumerate(zip(x_span, Nx))
            ]
        case _:
            raise ValueError("Unexpected combinattion of arguments")
    return pyclaw.Domain(dim)


def apply_initial_condition(
    state: pyclaw.State,
    ic_factory: Callable[
        [Float[np.ndarray, " *x_grid"]], Float[np.ndarray, " *x_grid"]
    ],
) -> None:
    grid_centers = grid_centers_from_state(state)
    grid = np.meshgrid(*grid_centers)
    state.q[0] = np.asarray(ic_factory(*grid))
