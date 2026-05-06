from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from context_flux_no.custom_types import FloatScalar


class AbstractTrainLoss[M: eqx.Module, B: PyTree[Array]](Protocol):
    def __call__(
        self, model: M, batch: B, args: Any, key: PRNGKeyArray
    ) -> tuple[FloatScalar, dict[str, Array]]:
        pass


class OneStepLoss[M: eqx.Module, B: PyTree[Array]](eqx.Module):
    def __call__(
        self, model: M, batch: B, args: Any, key: PRNGKeyArray
    ) -> tuple[FloatScalar, dict[str, Array]]:
        u: Float[Array, "batch time dim ..."] = batch
        u0, u1 = u[:, :-1], u[:, -1]

        keys = jax.random.split(key, u0.shape[0])
        u1_pred: Float[Array, "batch dim ..."] = eqx.filter_vmap(
            lambda u_, key_: model(u_, args, key=key_)
        )(u0, keys)[0]
        return jnp.mean((u1 - u1_pred) ** 2), dict()


class DenoisingOneStepLoss[M: eqx.Module, B: PyTree[Array]](eqx.Module):
    noise_scale: float = eqx.field(static=True, default=5e-5)

    def __call__(
        self, model: M, batch: B, args: Any, key: PRNGKeyArray
    ) -> tuple[FloatScalar, dict[str, Array]]:
        u: Float[Array, "batch time dim ..."] = batch
        u0, u1 = u[:, :-1], u[:, -1]
        u0_norm = jnp.sqrt(jnp.sum(u0**2, axis=tuple(range(3, u0.ndim)), keepdims=True))

        key, key_noise = jax.random.split(key)
        noise = self.noise_scale * u0_norm * jax.random.normal(key_noise, u0.shape)

        keys = jax.random.split(key, u0.shape[0])
        u1_pred: Float[Array, "batch dim ..."] = eqx.filter_vmap(
            lambda u_, key_: model(u_, args, key=key_)
        )(u0 + noise, keys)[0]
        return jnp.mean((u1 - u1_pred) ** 2), dict()


class PushforwardOneStepLoss[M: eqx.Module, B: PyTree[Array]](eqx.Module):
    def __call__(
        self, model: M, batch: B, args: Any, key: PRNGKeyArray
    ) -> tuple[FloatScalar, dict[str, Array]]:
        u: Float[Array, "batch time dim ..."] = batch
        # Different from other methods; thus to use this method dataloader segment size
        # must be set to be one larger usual
        u0_prev, u1 = u[:, :-2], u[:, -1]

        key_prev, key = jax.random.split(key)
        u1_prev = jax.lax.stop_gradient(
            eqx.filter_vmap(lambda u_, key_: model(u_, args, key=key_))(
                u0_prev, jax.random.split(key_prev, u0_prev.shape[0])
            )[0]
        )

        u0 = jnp.concatenate((u0_prev[:, 1:], jnp.expand_dims(u1_prev, axis=1)), axis=1)
        keys = jax.random.split(key, u0.shape[0])
        u1_pred: Float[Array, "batch dim ..."] = eqx.filter_vmap(
            lambda u_, key_: model(u_, args, key=key_)
        )(u0, keys)[0]
        return jnp.mean((u1 - u1_pred) ** 2), dict()
