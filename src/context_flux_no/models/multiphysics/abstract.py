import abc
from typing import Any, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from context_flux_no.utils import num_parameters


# The solver that an `AbstractGeneratedSolverOperator` generates from a context.
SolverT = TypeVar("SolverT")


class AbstractMultiphysicsOperator(eqx.Module):
    """Abstract class for multiphysics neural operator models that accept contiguous
    space-time blocks as input and return the predicted solution for the next time
    step.
    """

    num_spatial_dims: eqx.AbstractVar[int]

    def num_parameters(self) -> int:
        return num_parameters(self)

    @abc.abstractmethod
    def __call__(
        self,
        u: Float[Array, "time channels *grids"],
        args: Any,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> tuple[Float[Array, "channels *grids"], Any]:
        """Return the prediction for the next time step and auxiliary outputs."""
        pass

    def rollout(
        self,
        u: Float[Array, "time channels *grids"],
        args: Any,
        num_steps: int,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> tuple[Float[Array, "{num_steps} channels *grids"], Any]:
        """Roll out future predictions autoregressively."""

        keys = jax.random.split(key, num_steps) if key is not None else None

        def _scan_fn(
            u_in: Float[Array, "time channels *grids"],
            key_: PRNGKeyArray | None,
        ) -> tuple[
            Float[Array, "time channels *grids"],
            tuple[Float[Array, "channels *grids"], Any],
        ]:
            u_out, aux = self(
                u_in,
                args,
                key=key_,
                inference=inference,
            )

            u_in_next = jnp.concatenate(
                (
                    u_in[1:],
                    jnp.expand_dims(u_out, axis=0),
                ),
                axis=0,
            )

            return u_in_next, (u_out, aux)

        return jax.lax.scan(
            _scan_fn,
            u,
            xs=keys,
            length=num_steps,
        )[1]


class AbstractGeneratedSolverOperator(AbstractMultiphysicsOperator, Generic[SolverT]):
    """Abstract class for operators whose step is a solver generated from the context.

    A step of such an operator has two stages: the context determines a solver, and
    the solver advances the last state. `generate_solver` and `step` expose the two
    stages, so that the context and the advanced state can be chosen separately and
    one solver can advance several steps (ICLR 2027 rebuttal plan, actions 3a, 3e and
    7c). `step(generate_solver(u), u[-1], args)` equals `__call__(u, args)[0]` for
    every subclass.
    """

    @abc.abstractmethod
    def generate_solver(
        self,
        context: Float[Array, "time channels *grids"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> SolverT:
        """Return the solver that a context determines.

        **Arguments:**

        - `context`: The frames that the encoder reads.
        """

    @abc.abstractmethod
    def step(
        self,
        solver: SolverT,
        u: Float[Array, " channels *grids"],
        args: tuple[float, ...],
    ) -> Float[Array, " channels *grids"]:
        """Advance one state by one step with a given solver.

        **Arguments:**

        - `solver`: The solver, as `generate_solver` returns it.
        - `u`: The state to advance.
        - `args`: The time step followed by the grid spacing of every spatial dimension.
        """
