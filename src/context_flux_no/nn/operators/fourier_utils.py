import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def valid_frequency_inds(
    rfft_shape: tuple[int, ...], max_frequency_modes: tuple[int, ...]
) -> tuple[Int[Array, "..."], ...]:
    """Given the output shape of rfft transform and the maximum frequencies of modes to
    retain along each axis, return the corresponding array indices.

    If frequency mode = 0 for some axis, the zero frequency Fourier mode index will be
    returned.
    If the frequency mode for some axis exceeds the Nyquist frequency (shape//2 for all
    axes except the last; shape for the last axis), the entire axis will be indexed.

    This function is useful when implementing multi-dimensional Fourier Neural Operator
    and its variants.

    To better understand how the slice indices are formed, consult the documentation for
    numpy.fft.fftfreq and numpy.fft.rfftfreq.
    """
    if len(rfft_shape) != len(max_frequency_modes):
        raise ValueError(
            """rfft_shape and max_frequency_modes must have identical lengths"""
        )

    slices = []
    for n, f_max in zip(rfft_shape[:-1], max_frequency_modes[:-1]):
        if f_max >= n // 2:
            slices.append(jnp.r_[0:n])
        else:
            slices.append(jnp.r_[0 : f_max + 1, n - f_max : n])
    if max_frequency_modes[-1] >= rfft_shape[-1]:
        slices.append(jnp.r_[0 : rfft_shape[-1]])
    else:
        slices.append(jnp.r_[0 : max_frequency_modes[-1] + 1])
    return jnp.ix_(*slices)


def get_grid(
    x: Float[Array, " channels *grids"],
) -> Float[Array, " num_spatial_dims *grids"]:
    _, *num_grids = x.shape
    grids = jnp.meshgrid(*[jnp.linspace(0, 1, n) for n in num_grids], indexing="ij")
    return jnp.stack(grids, axis=0)


def append_grid_channels(
    x: Float[Array, " channels *grids"],
) -> Float[Array, " channels+num_spatial_dims *grids"]:
    with jax.ensure_compile_time_eval():
        grid = get_grid(x)
    return jnp.concatenate((x, grid), axis=0)
