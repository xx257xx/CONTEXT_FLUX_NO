import abc
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from context_flux_no.utils import num_parameters


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

    @abc.abstractmethod
    def encode_context(
        self,
        u: Float[Array, "time channels *grids"],
        args: Any,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Float[Array, "embedding_dim"]:
        """Return the context embedding used to condition the operator."""
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