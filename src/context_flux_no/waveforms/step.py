import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


def generate_periodic_random_step_function_1d(
    x: Float[Array, " Nx"],
    key: PRNGKeyArray,
    *,
    num_jumps_min: int = 1,
    num_jumps_max: int = 5,
    value_min: float = -1.0,
    value_max: float = 1.0,
) -> Float[Array, " Nx"]:
    """
    Generate a periodic random step function on a 1D grid.

    The function is piecewise constant on the periodic domain, so the left and right
    boundaries are connected.

    Args:
        x: 1D equispaced grid, shape (Nx,)
        key: JAX random key
        num_jumps_min: minimum number of jump points
        num_jumps_max: maximum number of jump points
        value_min, value_max: range of step heights

    Returns:
        u0: shape (Nx,)
    """
    Nx = x.shape[0]
    idx = jnp.arange(Nx)

    key_n, key_bp, key_val, key_shift = jax.random.split(key, 4)

    # number of jumps on the periodic domain
    num_jumps = jax.random.randint(
        key_n, shape=(), minval=num_jumps_min, maxval=num_jumps_max + 1
    )

    # choose jump locations among grid indices
    perm = jax.random.permutation(key_bp, idx)
    jump_idx = jnp.sort(perm[:num_jumps])

    # step values for each interval
    values = jax.random.uniform(
        key_val, shape=(num_jumps,), minval=value_min, maxval=value_max
    )

    # periodic random shift so that boundary x[0] is not special
    shift = jax.random.randint(key_shift, shape=(), minval=0, maxval=Nx)
    idx_shifted = (idx + shift) % Nx

    # count how many jump points are <= each shifted index
    # region label in {0,1,...,num_jumps-1}
    counts = jnp.sum(idx_shifted[:, None] >= jump_idx[None, :], axis=1)
    region = counts % num_jumps

    u0 = values[region]
    return u0


class PeriodicRandomStepFunction1D(eqx.Module):
    num_jumps_min: int = eqx.field(static=True, default=1)
    num_jumps_max: int = eqx.field(static=True, default=5)
    value_min: float = eqx.field(static=True, default=-1.0)
    value_max: float = eqx.field(static=True, default=1.0)

    def sample(self, x: Float[Array, " Nx"], key: PRNGKeyArray) -> Float[Array, " Nx"]:
        return jnp.expand_dims(
            generate_periodic_random_step_function_1d(
                x,
                key,
                num_jumps_min=self.num_jumps_min,
                num_jumps_max=self.num_jumps_max,
                value_min=self.value_min,
                value_max=self.value_max,
            ),
            axis=0,
        )


class MultichannelWaveform(eqx.Module):
    waveforms: tuple

    def __init__(self, waveforms):
        self.waveforms = tuple(waveforms)

    @property
    def channels(self) -> int:
        return len(self.waveforms)

    def sample(self, x: Float[Array, " Nx"], key: PRNGKeyArray) -> Float[Array, "C Nx"]:
        keys = jax.random.split(key, self.channels)
        return jnp.concatenate(
            [wave.sample(x, k) for wave, k in zip(self.waveforms, keys)], axis=0
        )
