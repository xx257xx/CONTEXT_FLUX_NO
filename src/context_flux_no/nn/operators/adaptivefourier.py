from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from equinox._misc import default_floating_dtype
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ...custom_types import FloatArray
from ..misc import apply_componentwise, to_complex_dtype, to_ntuple
from .fourier_utils import valid_frequency_inds


class AdaptiveFourier(eqx.Module):
    """Implementation of the Adaptive Fourier Neural Operator [1, 2].

    Adapted from the original PyTorch implementation at https://github.com/NVlabs/AFNO-transformer/tree/master.

    [1] J. Guibas et al. Efficient Token Mixing for Transformers via Adaptive Fourier
    Neural Operators. ICLR (2021).
    [2] Z. Hao et al. DPOT: Auto-Regressive Denoising Operator Transformer for
    Large-Scale PDE Pre-Training. ICML (2024).
    """

    num_spatial_dims: int = eqx.field(static=True)
    per_block_mlp: eqx.nn.MLP
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    max_frequency_modes: tuple[int, ...] = eqx.field(static=True)
    num_blocks: int = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        max_frequency_modes: int | tuple[int, ...],
        hidden_channels: int,
        num_blocks: int,
        activation: Callable[[FloatArray], FloatArray] = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        # Check channels
        if any(
            c % num_blocks != 0 for c in (in_channels, out_channels, hidden_channels)
        ):
            raise ValueError(
                """in_channels, out_channels, hidden_channels nust all be divisible by 
                num_blocks"""
            )
        if out_channels != in_channels:
            raise NotImplementedError(
                "The case for out_channels != in_channels is not yet handled"
            )

        # Parse max_frequency_modes
        max_frequency_modes = to_ntuple(max_frequency_modes, num_spatial_dims)

        # Set dtype
        dtype = default_floating_dtype() if dtype is None else dtype

        # Make MLPs to process blocks in parallel
        @eqx.filter_vmap
        def make_per_block_MLPs(key):
            return eqx.nn.MLP(
                in_size=in_channels // num_blocks,
                out_size=out_channels // num_blocks,
                width_size=hidden_channels // num_blocks,
                depth=1,
                activation=apply_componentwise(activation),
                dtype=to_complex_dtype(dtype),
                key=key,
            )

        self.per_block_mlp = make_per_block_MLPs(jax.random.split(key, num_blocks))

        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_frequency_modes = max_frequency_modes
        self.num_blocks = num_blocks

    @property
    def in_blocksize(self) -> int:
        return self.in_channels // self.num_blocks

    @property
    def out_blocksize(self) -> int:
        return self.out_channels // self.num_blocks

    def __call__(
        self, x: Float[Array, " in_channels *grids"]
    ) -> Float[Array, " out_channels *grids"]:
        x_fft = jnp.fft.rfftn(x, axes=tuple(range(1, x.ndim)))

        with jax.ensure_compile_time_eval():
            freq_mask = valid_frequency_inds(x_fft.shape[1:], self.max_frequency_modes)
        x_fft_masked = x_fft[:, *freq_mask]
        x_fft_masked = rearrange(
            x_fft_masked, "(n b_in) ... -> n b_in ...", n=self.num_blocks
        )

        @eqx.filter_vmap
        def apply_mlp_per_block(mlp, z: Complex[Array, " in_blocksize *freqs"]):
            in_shape = z.shape
            out_shape = (self.out_blocksize,) + in_shape[1:]

            z = jnp.reshape(z, shape=(in_shape[0], -1))
            z_out = eqx.filter_vmap(mlp, in_axes=-1, out_axes=-1)(z)
            return jnp.reshape(z_out, shape=out_shape)

        y_fft_masked = apply_mlp_per_block(self.per_block_mlp, x_fft_masked)
        y_fft_masked = rearrange(y_fft_masked, "n b_out ... -> (n b_out) ...")

        y_fft = jnp.zeros_like(x_fft).at[:, *freq_mask].set(y_fft_masked)
        y = jnp.fft.irfftn(y_fft, s=x.shape[1:])

        # If out_channels is not in_channels, residual connection will not make sense
        return y + x
