from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox._misc import default_floating_dtype
from equinox.nn._misc import default_init, named_scope
from jaxtyping import Array, Complex, Float, Int, PRNGKeyArray

from ..misc import to_complex_dtype, to_ntuple


class SpectralConv(eqx.Module):
    """General N-dimensional spectral convolution layer, which can be used to construct
    N-dimensional Fourier layers for (typically) neural operator applications."""

    num_spatial_dims: int = eqx.field(static=True)
    weight: Complex[Array, "out_channels in_channels frequency_modes"]
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    frequency_modes: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        frequency_modes: int | Sequence[int],
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        r"""**Arguments**

        - `num_spatial_dims`: Number of spatial dimensions.
            To build an N-dimensional Fourier Neural Operator, one requires
            `num_spatial_dims=N`.
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
        frequency_modes = to_ntuple(frequency_modes, num_spatial_dims)
        # The unusual shape of self.weight is due to the asymmetrical output shape of
        # jax.numpy.fft.fftn/ifftn
        wshape = (
            out_channels,
            in_channels,
            *[2 * i for i in frequency_modes[:-1]],
            frequency_modes[-1],
        )
        dtype = default_floating_dtype() if dtype is None else dtype

        # Initialization scheme identical to the original FNO code
        # Intriguingly, this is different from conventional initialization strategies
        # such as Xavier
        lim = 1 / (in_channels * out_channels)
        self.weight = default_init(key, wshape, to_complex_dtype(dtype), lim)

        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.frequency_modes = frequency_modes

    @named_scope("nn.SpectralConv")
    def __call__(
        self, v: Float[Array, " in_channels *grids"], *, key: PRNGKeyArray | None = None
    ) -> Float[Array, " out_channels *grids"]:
        """**Arguments**

        - `v`: The input, corresponding to 1D function values sampled from a equi-spaced
            grid. Should be a JAX array of shape `(in_channels, *grids)`.
            `grids` corresponds to an N-tuple of integers corresponding to the number of
            grid points per spatial dimension. Thus `len(grids)=self.num_spatial_dims`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        For each frequency of the frequency_modes used in the layer, the spectral
        component corresponding to each input channel is linearly mixed to produce the
        spectral components of the output channels.
        """
        n_grids = v.shape[1:]
        # TODO: rfftn axes argument seems wrong: should verify
        v_fft: Complex[Array, "in_channels grids"] = jnp.fft.rfftn(v, axes=(-1,))
        v_fft_trunc = v_fft[:, *self.frequency_mask(n_grids)]
        out_fft_trunc: Complex[Array, "out_channels frequency_modes"] = jnp.einsum(
            "ij...,j...->i...",
            self.weight,
            v_fft_trunc,
        )
        out_fft = jnp.zeros_like(
            v_fft, shape=(self.out_channels, *n_grids[:-1], n_grids[-1] // 2 + 1)
        )
        out_fft = out_fft.at[:, *self.frequency_mask(n_grids)].set(out_fft_trunc)
        out = jnp.fft.irfftn(out_fft, s=n_grids)
        return out

    def frequency_mask(
        self, grid_shape: tuple[int, ...]
    ) -> tuple[Int[Array, "..."], ...]:
        *n_grids, _ = grid_shape
        *n_modes, n_mode_last = self.frequency_modes
        return jnp.ix_(
            *[jnp.r_[:m, g - m : g] for (g, m) in zip(n_grids, n_modes)],
            jnp.r_[0:n_mode_last],
        )


class SpectralConv1D(SpectralConv):
    """As [`nn.SpectralConv`][] with `num_spatial_dims=1`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        frequency_modes: int | Sequence[int],
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(
            num_spatial_dims=1,
            in_channels=in_channels,
            out_channels=out_channels,
            frequency_modes=frequency_modes,
            dtype=dtype,
            key=key,
        )


class SpectralConv2D(SpectralConv):
    """As [`nn.SpectralConv`][] with `num_spatial_dims=2`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        frequency_modes: int | Sequence[int],
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            frequency_modes=frequency_modes,
            dtype=dtype,
            key=key,
        )


class SpectralConv3D(SpectralConv):
    """As [`nn.SpectralConv`][] with `num_spatial_dims=3`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        frequency_modes: int | Sequence[int],
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(
            num_spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            frequency_modes=frequency_modes,
            dtype=dtype,
            key=key,
        )


class Fourier(eqx.Module):
    """N-dimensional Fourier layer, which is commonly stacked to create Fourier Neural
    Operators."""

    num_spatial_dims: int
    spectral_conv: SpectralConv
    linear_transform: eqx.nn.Conv
    activation: Callable
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    frequency_modes: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        frequency_modes: int | Sequence[int],
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        r"""
        **Arguments**

        - `num_spatial_dims`: Number of spatial dimensions.
        - `in_channels`: Number of channels (dimensions) of the input functions. The
            input to the layer should be a vector of shape `(in_channels, grids)`,
            where `grids` is the number of equi-spaced grid points the function was
            sampled on.
        - `out_channels`: Number of channels (dimensions) of the output functions. The
            output from the layer will be a vector of shape `(out_channels, grids)`.
        - `frequency_modes`: Number of frequency modes per dimensoin that will be used
            in the layer.
            For an equi-spaced grid of $n$(=`grids`) points and width $L$, this
            corresponds to using $\lambda_j=j/2L, j=0, 1, \dots frequency_modes$
            spatial frequency values.
        - `dtype`: The dtype to use for the weight and the bias in this layer.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialization. (Keyword only argument.)
        """
        frequency_modes = to_ntuple(frequency_modes, num_spatial_dims)
        dtype = default_floating_dtype() if dtype is None else dtype
        skey, lkey = jax.random.split(key, 2)

        self.spectral_conv = SpectralConv(
            num_spatial_dims,
            in_channels,
            out_channels,
            frequency_modes,
            dtype,
            key=skey,
        )
        self.linear_transform = eqx.nn.Conv(
            num_spatial_dims, in_channels, out_channels, 1, key=lkey
        )

        self.num_spatial_dims = num_spatial_dims
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.frequency_modes = frequency_modes

    @named_scope("nn.Fourier")
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


class Fourier1D(Fourier):
    """As [`nn.Fourier`][] with `num_spatial_dims=1`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        frequency_modes: int | Sequence[int],
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(
            num_spatial_dims=1,
            in_channels=in_channels,
            out_channels=out_channels,
            frequency_modes=frequency_modes,
            activation=activation,
            dtype=dtype,
            key=key,
        )


class Fourier2D(Fourier):
    """As [`nn.Fourier`][] with `num_spatial_dims=2`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        frequency_modes: int | Sequence[int],
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            frequency_modes=frequency_modes,
            activation=activation,
            dtype=dtype,
            key=key,
        )


class Fourier3D(Fourier):
    """As [`nn.Fourier`][] with `num_spatial_dims=3`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        frequency_modes: int | Sequence[int],
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(
            num_spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            frequency_modes=frequency_modes,
            activation=activation,
            dtype=dtype,
            key=key,
        )
