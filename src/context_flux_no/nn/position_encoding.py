import abc

import equinox as eqx
import jax.numpy as jnp
from equinox.nn._misc import named_scope
from jaxtyping import Array, Float, PRNGKeyArray


class AbstractPositionEncoding(eqx.Module):
    num_spatial_dims: eqx.AbstractVar[int]

    @abc.abstractmethod
    def __call__(
        self, x: Float[Array, " channels *spatial_dims"]
    ) -> Float[Array, " channels *spatial_dims"]:
        pass


class LearnedPositionEncoding(AbstractPositionEncoding):
    encodings: Float[Array, "*shapes embedding_dim"]

    def __init__(self, spatial_dims: tuple[int, ...], channels: int):
        self.encodings = jnp.zeros(
            (*spatial_dims, channels),
        )

    def __call__(
        self, x: Float[Array, " channels *spatial_dims"]
    ) -> Float[Array, " channels *spatial_dims"]:
        if x.shape != self.encodings.shape:
            raise ValueError(
                """Input array shape does not match the shape of the learnable position 
                encodings."""
            )
        return x + self.encodings

    @property
    def num_spatial_dims(self) -> int:
        return self.encodings.ndim - 1


# TODO: Incorporate into the AbstractPositionEncoding type hierarchy
class SineCosinePosEncoding2D(eqx.Module):
    """Fixed 2D sine-cosine encoding for the simplified ViT, which was presented in
    L. Beyer et al, Better plain ViT baselines for ImageNet-1k, arXiv:2205.01580v1
    (2022).

    The code follows that of the original implmentation at
    https://github.com/google-research/big_vision/blob/main/big_vision/models/vit.py"""

    encoding_dim: int = eqx.field(static=True)
    temperature: float = eqx.field(static=True)

    def __init__(
        self, encoding_dim: int, temperature: float = 10000.0, *, key: PRNGKeyArray
    ):
        del key
        if encoding_dim % 4 != 0:
            raise ValueError("Encoding dimension must be a multiple of 4.")
        self.encoding_dim = encoding_dim
        self.temperature = temperature

    @named_scope("nn.SineCosinePosEncoding2D")
    def __call__(
        self, row: int, col: int, *, key: PRNGKeyArray | None = None
    ) -> Float[Array, "row col encoding_dim"]:
        y, x = jnp.mgrid[:row, :col]
        k = 1 / jnp.logspace(0.0, 1.0, self.encoding_dim // 4, base=self.temperature)

        kx: Float[Array, "row col encoding_dim//4"] = jnp.einsum("...,d->...d", x, k)
        ky: Float[Array, "row col encoding_dim//4"] = jnp.einsum("...,d->...d", y, k)
        pos_embed: Float[Array, "row col encoding_dim"] = jnp.concatenate(
            [jnp.sin(kx), jnp.cos(kx), jnp.sin(ky), jnp.cos(ky)], axis=-1
        )
        return pos_embed
