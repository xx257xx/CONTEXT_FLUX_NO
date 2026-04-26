from math import prod

import jax.numpy as jnp
from jaxtyping import Array, Float


def L2_norm(
    u: Float[Array, " channels *spatial_dims"], dxs: tuple[float, ...] | None = None
) -> Float[Array, ""]:
    # Sum over channels for inner product, sum over spatial_dims for integration
    norm_squared = jnp.sum(jnp.abs(u) ** 2)
    if dxs is not None:
        norm_squared = norm_squared * prod(dxs)
    return jnp.sqrt(norm_squared)


def L_infty_norm(u: Float[Array, " channels *spatial_dims"]) -> Float[Array, ""]:
    return jnp.max(jnp.abs(u))


def relative_L2_error(
    u_pred: Float[Array, " channels *spatial_dims"],
    u_data: Float[Array, " channels *spatial_dims"],
) -> Float[Array, ""]:
    """Compute the relative l2 error between the predicted and data fields."""
    return L2_norm(u_pred - u_data) / L2_norm(u_data)


def relative_L_infty_error(
    u_pred: Float[Array, " channels *spatial_dims"],
    u_data: Float[Array, " channels *spatial_dims"],
) -> Float[Array, ""]:
    """Compute the relative l2 error between the predicted and data fields."""
    return L_infty_norm(u_pred - u_data) / L_infty_norm(u_data)
