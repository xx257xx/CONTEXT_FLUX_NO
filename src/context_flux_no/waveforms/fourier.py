import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class TruncatedFourier1D(eqx.Module):
    an: Float[Array, " K"]
    bn: Float[Array, " K"]
    a0: Float[Array, ""] = eqx.field(default_factory=lambda: jnp.array(0.0))
    L: float = eqx.field(static=True, default=1.0)

    def __call__(self, x: Float[Array, " *shape"]):
        kx: Float[Array, "*shape K"] = jnp.expand_dims(x, axis=-1) * self.wavenumbers
        cos_kx, sin_kx = jnp.cos(kx), jnp.sin(kx)
        return self.a0 + jnp.mean(
            self.an * cos_kx + self.bn * sin_kx, axis=-1
        ) / jnp.sqrt(2)

    @property
    def num_modes(self) -> int:
        return len(self.an)

    @property
    def wavenumbers(self) -> Float[Array, " K"]:
        return jnp.arange(1, self.num_modes + 1) * 2 * jnp.pi / self.L

    @classmethod
    def with_uniform_rand_coeffs(
        cls,
        num_modes: int,
        L: float = 1.0,
        coeff_range: tuple[float, float] = (1, 1),
        offset_range: tuple[float, float] | None = None,
        *,
        key: PRNGKeyArray = jax.random.PRNGKey(0),
    ):
        key_coeff, key_offset = jax.random.split(key, 2)
        an_bn = jax.random.uniform(
            key_coeff,
            shape=(2, num_modes),
            minval=coeff_range[0],
            maxval=coeff_range[1],
        )
        if offset_range is None:
            a0 = jnp.array(0.0)
        else:
            a0 = jax.random.uniform(
                key_offset, minval=offset_range[0], maxval=offset_range[1]
            )
        return cls(*an_bn, a0, L)
