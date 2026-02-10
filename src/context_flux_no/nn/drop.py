import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class DropPath(eqx.Module):
    """JAX implementation of DropPath.

    Adapted from eqxvision and timm.
    Effectively dropping a sample from the call.
    Often used inside a network along side a residual connection.
    Equivalent to `torchvision.stochastic_depth`."""

    p: float
    inference: bool
    scale_by_keep: bool = eqx.field(static=True)
    mode: str = eqx.field(static=True)

    def __init__(
        self,
        p: float = 0.0,
        inference: bool = False,
        *,
        scale_by_keep: bool = True,
        mode="global",
    ):
        """**Arguments:**

        - `p`: The probability to drop a sample entirely during forward pass
        - `inference`: Defaults to `False`. If `True`, then the input is returned
            unchanged. This may be toggled with `equinox.tree_inference`
        - `mode`: Can be set to `global` or `local`. If `global`, the whole input is
            dropped or retained. If `local`, then the decision on each input unit is
            computed independently. Defaults to `global`

        !!! note
            For `mode = local`, an input `(channels, dim_0, dim_1, ...)` is reshaped
            and transposed to `(channels, dims).transpose()`. For each `dim x channels`
            element, the decision to drop/keep is made independently.
        """
        self.p = p
        self.inference = inference
        self.scale_by_keep = scale_by_keep
        self.mode = mode

    def __call__(
        self,
        x,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Array:
        """**Arguments:**

        - `x`: An any-dimensional JAX array to drop
        - `key`: A `jax.random.PRNGKey` used to provide randomness for calculating
            which elements to dropout. (Keyword only argument.)
        - `inference`: As per [`equinox.nn.Dropout.__init__`][]. If `True` or
            `False` then it will take priority over `self.inference`. If `None`
            then the value from `self.inference` will be used.
        """
        if inference is None:
            inference = self.inference
        if isinstance(self.p, (int, float)) and self.p == 0:
            inference = True

        if inference:
            return x
        elif key is None:
            raise RuntimeError(
                """DropPath requires a key when running in non-deterministic mode. Did 
                you mean to enable inference?"""
            )
        else:
            keep_prob = 1 - jax.lax.stop_gradient(self.p)
            if self.mode == "global":
                noise = jax.random.bernoulli(key, p=keep_prob)
            else:
                noise = jnp.expand_dims(
                    jax.random.bernoulli(key, p=keep_prob, shape=[x.shape[0]]).reshape(
                        -1
                    ),
                    axis=[i for i in range(1, len(x.shape))],
                )
            if self.scale_by_keep:
                noise /= keep_prob
            return x * noise
