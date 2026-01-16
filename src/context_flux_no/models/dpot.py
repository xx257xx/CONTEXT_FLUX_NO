import equinox as eqx
import jax
import jax.numpy as jnp
from equinox._misc import default_floating_dtype
from jaxtyping import Array, Float, PRNGKeyArray


class TimeAggregator(eqx.Module):
    fourier_freqs: Float[Array, " channels"]
    weights: Float[Array, "timesteps channels channels"]

    timesteps: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)

    def __init__(self, timesteps: int, channels: int, dtype=None, *, key: PRNGKeyArray):
        dtype = default_floating_dtype() if dtype is None else dtype
        self.fourier_freqs = jnp.logspace(-10, 10, channels, base=2, dtype=dtype)

        scale = 1 / (timesteps * channels**0.5)
        self.weights = scale * jax.random.normal(
            key, shape=(timesteps, channels, channels), dtype=dtype
        )
        self.timesteps = timesteps
        self.channels = channels

    def __call__(
        self, u: Float[Array, "time channels *grids"]
    ) -> Float[Array, " channels *grids"]:
        with jax.ensure_compile_time_eval():
            t = jnp.linspace(0, 1, u.shape[0])
        fourier_features: Float[Array, "time channels"] = jnp.cos(
            jnp.outer(t, self.fourier_freqs)
        )
        return jnp.einsum("tij,ti...,ti->j...", self.weights, u, fourier_features)
