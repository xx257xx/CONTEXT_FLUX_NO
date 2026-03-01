import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox._misc import default_floating_dtype
from equinox.nn._misc import named_scope
from jaxtyping import Array, Float, PRNGKeyArray


class AbstractPositionEncoding(eqx.Module):
    num_spatial_dims: eqx.AbstractVar[int]

    @abc.abstractmethod
    def __call__(
        self, u: Float[Array, " channels *spatial_dims"]
    ) -> Float[Array, " channels *spatial_dims"]:
        pass


class LearnedPositionEncoding(AbstractPositionEncoding):
    encodings: Float[Array, " channels *spatial_dims"]

    def __init__(
        self,
        channels: int,
        spatial_dims: tuple[int, ...],
        init_scale: float,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        if dtype is None:
            dtype = default_floating_dtype()

        self.encodings = (
            jax.random.normal(
                key=key,
                shape=(
                    channels,
                    *spatial_dims,
                ),
                dtype=dtype,
            )
            * init_scale
        )

    def __call__(
        self, u: Float[Array, " channels *spatial_dims"]
    ) -> Float[Array, " channels *spatial_dims"]:
        if u.shape != self.encodings.shape:
            raise ValueError(
                """Input array shape does not match the shape of the learnable position 
                encodings."""
            )
        return u + self.encodings

    @property
    def num_spatial_dims(self) -> int:
        return self.encodings.ndim - 1


# TODO: Incorporate into the AbstractPositionEncoding type hierarchy
class SineCosinePosEncoding2D(AbstractPositionEncoding):
    """Fixed 2D sine-cosine encoding for the simplified ViT, which was presented in
    L. Beyer et al, Better plain ViT baselines for ImageNet-1k, arXiv:2205.01580v1
    (2022).

    The code follows that of the original implmentation at
    https://github.com/google-research/big_vision/blob/main/big_vision/models/vit.py"""

    channels: int = eqx.field(static=True)
    temperature: float = eqx.field(static=True)

    def __init__(
        self, channels: int, temperature: float = 10000.0, *, key: PRNGKeyArray
    ):
        del key
        if channels % 4 != 0:
            raise ValueError("Encoding dimension must be a multiple of 4.")
        self.channels = channels
        self.temperature = temperature

    def make_encodings(
        self, row: int, col: int
    ) -> Float[Array, "{self.channels} {row} {col}"]:
        y, x = jnp.mgrid[:row, :col]
        k = 1 / jnp.logspace(0.0, 1.0, self.channels // 4, base=self.temperature)

        kx: Float[Array, "channels//4 row col"] = jnp.einsum("...,d->d...", x, k)
        ky: Float[Array, "channels//4 row col"] = jnp.einsum("...,d->d...", y, k)
        return jnp.concatenate(
            [jnp.sin(kx), jnp.cos(kx), jnp.sin(ky), jnp.cos(ky)], axis=0
        )

    @named_scope("nn.SineCosinePosEncoding2D")
    def __call__(
        self,
        u: Float[Array, " channels *spatial_dims"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " channels *spatial_dims"]:
        if u.ndim != 3:
            raise ValueError("Input array must be 3D, with shape (channels, row, col)")
        with jax.ensure_compile_time_eval():
            encodings = self.make_encodings(*u.shape[1:])
        return u + encodings
