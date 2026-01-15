from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.nn._misc import named_scope
from jaxtyping import Array, Float, PRNGKeyArray

from .misc import to_ntuple


class PatchEmbedding(eqx.Module):
    layers: tuple[eqx.nn.Conv, ...]

    num_spatial_dims: int = eqx.field(static=True)
    patch_size: tuple[int] = eqx.field(static=True)
    in_dim: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    final_activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        patch_size: int | Sequence[int],
        in_dim: int,
        embedding_dim: int,
        num_hidden: int = 0,
        hidden_dim: int = 32,
        activation: Callable = jax.nn.gelu,
        final_activation: Callable = lambda x: x,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        patch_size = to_ntuple(patch_size, num_spatial_dims)
        if num_patch_dims := len(patch_size) != num_spatial_dims:
            raise ValueError(
                f"""Given num_spatial_dims is incompatible with that of patch_size.
                Got {num_spatial_dims=} but patch_size has spatial dimensions 
                {num_patch_dims}"""
            )
        self.patch_size = patch_size

        # Largely inspired by the implementation of `equinox.nn.MLP`
        # https://github.com/patrick-kidger/equinox/blob/main/equinox/nn/_mlp.py#L88-L107
        keys = jax.random.split(key, num_hidden + 1)
        layers = []
        if num_hidden == 0:
            layers.append(
                eqx.nn.Conv(
                    num_spatial_dims,
                    in_dim,
                    embedding_dim,
                    self.patch_size,
                    self.patch_size,
                    dtype=dtype,
                    key=keys[0],
                )
            )
        else:
            layers.append(
                eqx.nn.Conv(
                    num_spatial_dims,
                    in_dim,
                    hidden_dim,
                    self.patch_size,
                    self.patch_size,
                    dtype=dtype,
                    key=keys[0],
                )
            )
            for i in range(num_hidden - 1):
                layers.append(
                    eqx.nn.Conv(
                        num_spatial_dims,
                        hidden_dim,
                        hidden_dim,
                        1,
                        1,
                        dtype=dtype,
                        key=keys[i],
                    )
                )
            layers.append(
                eqx.nn.Conv(
                    num_spatial_dims,
                    hidden_dim,
                    embedding_dim,
                    1,
                    1,
                    dtype=dtype,
                    key=keys[-1],
                )
            )
        self.layers = tuple(layers)
        self.num_spatial_dims = num_spatial_dims
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.activation = activation
        self.final_activation = final_activation

    def maybe_pad(
        self, x: Float[Array, " in_dim *grids"]
    ) -> Float[Array, " in_dim *grids_padded"]:
        """Given an array x, zero pad it at the end if necessary to make the image size
        an integer multiple of self.patch_size.

        Under jax.jit, this function should be compiled down to identity if no padding
        is needed; otherwise, it will become a single call to jax.numpy.pad."""
        # Compute pad widths and whether to pad
        with jax.ensure_compile_time_eval():
            pad_after = tuple(
                p - s % p if s % p != 0 else 0
                for (s, p) in zip(x.shape[1:], self.patch_size)
            )
            should_pad = any(s != 0 for s in pad_after)
        # Conditional evaluated during compile time
        if should_pad:
            pad_widths = [(0, 0)] + [(0, p) for p in pad_after]
            return jnp.pad(x, pad_widths)
        else:
            return x

    @named_scope("nn.PatchEmbedding")
    def __call__(
        self, x: Float[Array, " in_dim *grids"], *, key: PRNGKeyArray | None = None
    ) -> Float[Array, " embedding_dim *grids_patch"]:
        assert all(
            s % p == 0 for (s, p) in zip(x.shape[1:], self.patch_size)
        ), """Image shape mult be integer multiple of self.patch_size. 
        If not, call PatchEmbedding.maybe_pad first to pad the image to proper shape."""

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.final_activation(self.layers[-1](x))
