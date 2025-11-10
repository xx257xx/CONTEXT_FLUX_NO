from collections.abc import Callable, Sequence
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox._misc import default_floating_dtype
from jaxtyping import Array, Complex, Float, Int, PRNGKeyArray

from ..misc import to_complex_dtype, to_ntuple


class HyperFourier(eqx.Module):
    """Hypernetwork version of the N-dimensional Fourier layer.

    The complex weights for the spectral convolution operation is created
    using a MLP.

    The implementation closely follows that of nn.Fourier.
    """

    num_spatial_dims: int
    fourier_weight_net: eqx.nn.MLP
    linear_weight_net: eqx.nn.MLP
    linear_transform: eqx.nn.Conv
    activation: Callable
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    frequency_modes: tuple[int, ...] = eqx.field(static=True)
    fourier_weight_shape: tuple[int, ...] = eqx.field(static=True)
    linear_weight_shape: tuple[int, ...] = eqx.field(static=True)
    linear_bias_shape: tuple[int, ...] = eqx.field(static=True)
    hyper_in_dims: int = eqx.field(static=True)
    hyper_width: int = eqx.field(static=True)
    hyper_depth: int = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        frequency_modes: int | Sequence[int],
        hyper_in_dims: int,
        hyper_depth: int,
        hyper_width: int,
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        r"""**Arguments**

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
        w1key, w2key = jax.random.split(key, 2)

        self.fourier_weight_shape = (
            out_channels,
            in_channels,
            *[2 * i for i in frequency_modes[:-1]],
            frequency_modes[-1],
        )
        self.fourier_weight_net = eqx.nn.MLP(
            in_size=hyper_in_dims,
            out_size=prod(self.fourier_weight_shape),
            width_size=hyper_width,
            depth=hyper_depth,
            activation=activation,
            dtype=to_complex_dtype(dtype),
            key=w1key,
        )

        self.linear_weight_shape = (
            out_channels,
            in_channels,
        ) + (1,) * num_spatial_dims
        self.linear_bias_shape = (out_channels,) + (1,) * num_spatial_dims
        self.linear_weight_net = eqx.nn.MLP(
            in_size=hyper_in_dims,
            out_size=prod(self.linear_weight_shape) + prod(self.linear_bias_shape),
            width_size=hyper_width,
            depth=hyper_depth,
            activation=activation,
            dtype = dtype,
            key=w2key,
        )

        self.num_spatial_dims = num_spatial_dims
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.frequency_modes = frequency_modes
        self.hyper_in_dims = hyper_in_dims
        self.hyper_width = hyper_width
        self.hyper_depth = hyper_depth

    def spectral_conv(
        self,
        v: Float[Array, " in_channels *grids"],
        fourier_weights: Complex[Array, " {*self.fourier_weight_shape} *grids"],
    ) -> Float[Array, " out_channels *grids"]:
        """**Arguments**

        - `v`: The input, corresponding to 1D function values sampled from a equi-spaced
            grid. Should be a JAX array of shape `(in_channels, *grids)`.
            `grids` corresponds to an N-tuple of integers corresponding to the number of
            grid points per spatial dimension. Thus `len(grids)=self.num_spatial_dims`.

        For each frequency of the frequency_modes used in the layer, the spectral
        component corresponding to each input channel is linearly mixed to produce the
        spectral components of the output channels.
        """
        n_grids = v.shape[1:]
        v_fft: Complex[Array, "in_channels grids"] = jnp.fft.rfftn(v, axes=(-1,))
        v_fft_trunc = v_fft[:, *self.frequency_mask(n_grids)]
        out_fft_trunc: Complex[Array, "out_channels frequency_modes"] = jnp.einsum(
            "ij...,j...->i...",
            fourier_weights,
            v_fft_trunc,
        )
        out_fft = jnp.zeros_like(
            v_fft,
            shape=(self.out_channels, *n_grids[:-1], n_grids[-1] // 2 + 1),
        )
        out_fft = out_fft.at[:, *self.frequency_mask(n_grids)].set(out_fft_trunc)
        out = jnp.fft.irfftn(out_fft, s=n_grids)
        return out

    def linear_transform(
        self,
        v: Float[Array, " in_channels *grids"],
        linear_weight: Float[Array, " {*self.linear_weight_shape}"],
        linear_bias: Float[Array, " {*self.linear_weight_shape}"],
    ):
        v = jnp.expand_dims(v, axis=0)
        v = jax.lax.conv_general_dilated(
            lhs=v,
            rhs=linear_weight,
            window_strides=(1,) * self.num_spatial_dims,
            padding=[
                (0, 0),
            ]
            * self.num_spatial_dims,
        )
        v = jnp.squeeze(v, axis=0)
        return v + linear_bias

    def frequency_mask(
        self,
        grid_shape: tuple[int, ...],
    ) -> tuple[Int[Array, "..."], ...]:
        *n_grids, _ = grid_shape
        *n_modes, n_mode_last = self.frequency_modes
        return jnp.ix_(
            *[jnp.r_[:m, g - m : g] for (g, m) in zip(n_grids, n_modes, strict=False)],
            jnp.r_[0:n_mode_last],
        )

    def __call__(
        self,
        v: Float[Array, "in_channels grids"],
        hyper_input: Float[Array, " hyper_in_dim"],
        *,
        key: PRNGKeyArray | None = None,
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
        fourier_weights = jnp.reshape(
            self.fourier_weight_net(hyper_input),
            self.fourier_weight_shape,
        )
        linear_params_flat = self.linear_weight_net(hyper_input)
        linear_weight = jnp.reshape(
            linear_params_flat[: prod(self.linear_weight_shape)],
            self.linear_weight_shape,
        )
        linear_bias = jnp.reshape(
            linear_params_flat[prod(self.linear_weight_shape) :],
            self.linear_bias_shape,
        )
        v_out = self.spectral_conv(v, fourier_weights) + self.linear_transform(
            v, linear_weight, linear_bias
        )
        return self.activation(v_out)
