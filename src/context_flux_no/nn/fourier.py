from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox._misc import default_floating_dtype
from equinox.nn._misc import default_init, named_scope
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from .misc import to_complex_dtype


class SpectralConv1D(eqx.Module, strict=True):
    """1D Spectral convolution layer, which is used to construct 1D Fourier layers
    to be used for neural operators"""

    weight: Complex[Array, "out_channels in_channels frequency_modes"]
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    frequency_modes: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        frequency_modes: int,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments**

        - `in_channels`: Number of channels (dimensions) of the input functions. The
            input to the layer should be a vector of shape `(in_channels, grids)`,
            where `grids` is the number of equi-spaced grid points the function was
            sampled on.
        - `out_channels`: Number of channels (dimensions) of the output functions. The
            output from the layer will be a vector of shape `(out_channels, grids)`.
        - `frequency_modes`: Number of frequency modes that will be used in the layer.
            For an equi-spaced grid of $n$(=`grids`) points and width $L$, this
            corresponds to using $\lambda_j=j/2L, j=0, 1, \dots frequency_modes$
            spatial frequency values.
        - `dtype`: The dtype to use for the weight and the bias in this layer.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialization. (Keyword only argument.)

        Note that the parameters (`self.weight`) of this layer is complex, as they
        operate on the Fourier transform of the input data.
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        wshape = (out_channels, in_channels, frequency_modes)

        # Initialization scheme identical to the original FNO code
        # Intriguingly, this is different from conventional initialization strategies
        # such as Xavier
        lim = 1 / (in_channels * out_channels)
        self.weight = default_init(key, wshape, to_complex_dtype(dtype), lim)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.frequency_modes = frequency_modes

    @named_scope("nn.SpectralConv1D")
    def __call__(
        self, v: Float[Array, "in_channels grids"], *, key: PRNGKeyArray | None = None
    ) -> Float[Array, "out_channels grids"]:
        """**Arguments**

        - `v`: The input, corresponding to 1D function values sampled from a equi-spaced
            grid. Should be a JAX array of shape `(in_channels, grids)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        For each frequency of the frequency_modes used in the layer, the spectral
        component corresponding to each input channel is linearly mixed to produce the
        spectral components of the output channels.
        """
        n_grids = v.shape[-1]
        v_fft: Complex[Array, "in_channels grids"] = jnp.fft.rfft(v)
        out_fft_trunc: Complex[Array, "out_channels frequency_modes"] = jax.vmap(
            jnp.matvec, in_axes=-1, out_axes=-1
        )(self.weight, v_fft[:, 0 : self.frequency_modes])
        out_fft = jnp.zeros_like(v, shape=(self.out_channels, n_grids // 2 + 1))
        out_fft.at[:, 0 : self.frequency_modes].set(out_fft_trunc)

        out = jnp.fft.irfft(out_fft, n=n_grids)
        return out


class Fourier1D(eqx.Module, strict=True):
    """1D Fourier layer, which is commonly stacked to create Fourier Neural
    Operators."""

    spectral_conv: SpectralConv1D
    linear_transform: eqx.nn.Conv1d
    activation: Callable
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    frequency_modes: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        frequency_modes: int,
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """
        **Arguments**

        - `in_channels`: Number of channels (dimensions) of the input functions. The
            input to the layer should be a vector of shape `(in_channels, grids)`,
            where `grids` is the number of equi-spaced grid points the function was
            sampled on.
        - `out_channels`: Number of channels (dimensions) of the output functions. The
            output from the layer will be a vector of shape `(out_channels, grids)`.
        - `frequency_modes`: Number of frequency modes that will be used in the layer.
            For an equi-spaced grid of $n$(=`grids`) points and width $L$, this
            corresponds to using $\lambda_j=j/2L, j=0, 1, \dots frequency_modes$
            spatial frequency values.
        - `dtype`: The dtype to use for the weight and the bias in this layer.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialization. (Keyword only argument.)
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        skey, lkey = jax.random.split(key, 2)

        self.spectral_conv = SpectralConv1D(
            in_channels, out_channels, frequency_modes, dtype, key=skey
        )
        self.linear_transform = eqx.nn.Conv1d(in_channels, out_channels, 1, key=lkey)

        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.frequency_modes = frequency_modes

    @named_scope("nn.Fourier1D")
    def __call__(
        self, v: Float[Array, "in_channels grids"], *, key: PRNGKeyArray | None = None
    ) -> Float[Array, "out_channels grids"]:
        """**Arguments**

        - `v`: The input, corresponding to 1D function values sampled from a equi-spaced
            grid. Should be a JAX array of shape `(in_channels, grids)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        For each frequency of the frequency_modes used in the layer, the spectal
        component corresponding to each input channel is linearly mixed to produce the
        spectral components of the output channels.
        """
        v_out = self.spectral_conv(v) + self.linear_transform(v)
        return self.activation(v_out)
