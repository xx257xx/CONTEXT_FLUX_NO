import math
from collections.abc import Callable
from functools import partial
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from einops import rearrange
from equinox._misc import default_floating_dtype
from equinox.nn._misc import named_scope
from jaxtyping import Array, Bool, Float, PRNGKeyArray


def dot_product_attention_weights(
    query: Float[Array, "q_seq qk_size"],
    key: Float[Array, "kv_seq qk_size"],
    bias: Float[Array, "q_seq kv_seq"] | None = None,
    mask: Bool[Array, "q_seq kv_seq"] | None = None,
) -> Float[Array, "q_seq kv_seq"]:
    query = query / math.sqrt(query.shape[-1])
    logits = jnp.einsum("sd,Sd->sS", query, key)
    if mask is not None:
        if mask.shape != logits.shape:
            raise ValueError(
                f"mask must have shape (query_seq_length, "
                f"kv_seq_length)=({query.shape[0]}, "
                f"{key.shape[0]}). Got {mask.shape}."
            )
        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)
        logits = cast(Array, logits)
    if bias is not None:
        if bias.shape != logits.shape:
            raise ValueError(
                f"bias must have shape (query_seq_length, "
                f"kv_seq_length)=({query.shape[0]}, "
                f"{key.shape[0]}). Got {bias.shape}."
            )
        logits = logits + bias
    with jax.numpy_dtype_promotion("standard"):
        dtype = jnp.result_type(logits.dtype, jnp.float32)
    weights = jax.nn.softmax(logits.astype(dtype)).astype(logits.dtype)
    return weights


def dot_product_attention(
    query: Float[Array, "q_seq qk_size"],
    key_: Float[Array, "kv_seq qk_size"],
    value: Float[Array, "kv_seq v_size"],
    bias: Float[Array, "q_seq kv_seq"] | None = None,
    mask: Bool[Array, "q_seq kv_seq"] | None = None,
    dropout: eqx.nn.Dropout | None = None,
    *,
    key: PRNGKeyArray | None = None,
    inference: bool | None = None,
) -> Float[Array, "q_seq v_size"]:
    weights = dot_product_attention_weights(query, key_, bias, mask)
    if dropout is not None:
        weights = dropout(weights, key=key, inference=inference)
    attn = jnp.einsum("sS,Sd->sd", weights, value)
    return attn


_ProcessHeads = Callable[
    [
        Float[Array, "seq_length num_heads qk_size"],
        Float[Array, "seq_length num_heads qk_size"],
        Float[Array, "seq_length num_heads vo_size"],
    ],
    tuple[
        Float[Array, "seq_length num_heads qk_size"],
        Float[Array, "seq_length num_heads qk_size"],
        Float[Array, "seq_length num_heads vo_size"],
    ],
]
_Mask = Bool[Array, "q_seq kv_seq"] | Bool[Array, "num_heads q_seq kv_seq"]
_Bias = Float[Array, "q_seq kv_seq"] | Bool[Array, "num_heads q_seq kv_seq"]


class MultiheadAttention(eqx.Module):
    r"""
    An modified version of `equinox.nn.MultiheadAttention`.

    In contrast to `equinox.nn.MultiheadAttention`, supports QK-norm [1, 2] and the use
    of relative position encodings via the bias argument in MultiheadAttention.__call__.
    """

    query_proj: eqx.nn.Linear
    key_proj: eqx.nn.Linear
    value_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear | eqx.nn.Identity
    dropout: eqx.nn.Dropout
    query_norm: eqx.nn.LayerNorm | None
    key_norm: eqx.nn.LayerNorm | None

    num_heads: int = eqx.field(static=True)
    query_size: int = eqx.field(static=True)
    key_size: int = eqx.field(static=True)
    value_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    qk_size: int = eqx.field(static=True)
    vo_size: int = eqx.field(static=True)
    use_query_bias: bool = eqx.field(static=True)
    use_key_bias: bool = eqx.field(static=True)
    use_value_bias: bool = eqx.field(static=True)
    use_output_bias: bool = eqx.field(static=True)
    use_qk_norm: bool = eqx.field(static=True)
    use_output_proj: bool = eqx.field(static=True)

    def __init__(
        self,
        num_heads: int,
        query_size: int,
        key_size: int | None = None,
        value_size: int | None = None,
        output_size: int | None = None,
        qk_size: int | None = None,
        vo_size: int | None = None,
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_output_bias: bool = False,
        use_qk_norm: bool = False,
        use_output_proj: bool = True,
        dropout_p: float = 0.0,
        inference: bool = False,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        r"""**Arguments:**

        - `num_heads`: Number of parallel attention heads $h$.
        - `query_size`: Number of input channels for query $Q$.
        - `key_size`: Number of input channels for key $K$. Defaults to `query_size`.
        - `value_size`: Number of input channels for value $V$. Defaults to
            `query_size`.
        - `output_size`: Number of output channels. Defaults to `query_size`.
        - `qk_size`: Number of channels to compare query and key over, per head.
            Defaults to `query_size // num_heads`.
        - `vo_size`: Number of channels to compare attention-weighted value and output
            over, per head. Defaults to `query_size // num_heads`.
        - `use_query_bias`: Whether to use a bias term in the query projections.
        - `use_key_bias`: Whether to use a bias term in the key projections.
        - `use_value_bias`: Whether to use a bias term in the value projections.
        - `use_output_bias`: Whether to use a bias term in the output projection.
        - `use_qk_norm` :  Whether to apply QK-norm - that is, per-head LayerNorm on the
            query $Q$ and key $K$.
        - `use_output_proj` : Whether to use the output linear layer. When set to
            False, the concatenated output from the attention heads will be returned.
        - `dropout_p`: Dropout probability on attention weights.
        - `inference`: Whether to actually apply dropout at all. If `True` then dropout
            is not applied. If `False` then dropout is applied. This may be toggled
            with [`equinox.nn.inference_mode`][] or overridden during
            [`equinox.nn.MultiheadAttention.__call__`][].
        - `dtype`: The dtype to use for all trainable parameters in this layer.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        qkey, kkey, vkey, okey = jrandom.split(key, 4)

        if key_size is None:
            key_size = query_size
        if value_size is None:
            value_size = query_size
        if qk_size is None:
            qk_size = query_size // num_heads
        if vo_size is None:
            vo_size = query_size // num_heads
        if output_size is None:
            output_size = query_size

        self.query_proj = eqx.nn.Linear(
            query_size,
            num_heads * qk_size,
            use_bias=use_query_bias,
            dtype=dtype,
            key=qkey,
        )
        self.key_proj = eqx.nn.Linear(
            key_size, num_heads * qk_size, use_bias=use_key_bias, dtype=dtype, key=kkey
        )
        self.value_proj = eqx.nn.Linear(
            value_size,
            num_heads * vo_size,
            use_bias=use_value_bias,
            dtype=dtype,
            key=vkey,
        )
        if use_output_proj:
            self.output_proj = eqx.nn.Linear(
                num_heads * vo_size,
                output_size,
                use_bias=use_output_bias,
                dtype=dtype,
                key=okey,
            )
        else:
            self.output_proj = eqx.nn.Identity()
        self.dropout = eqx.nn.Dropout(dropout_p, inference=inference)

        # Initialize optional normalization layers
        if use_qk_norm:
            self.query_norm = eqx.nn.LayerNorm(qk_size)
            self.key_norm = eqx.nn.LayerNorm(qk_size)
        else:
            self.query_norm = None
            self.key_norm = None

        self.num_heads = num_heads
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.output_size = output_size
        self.qk_size = qk_size
        self.vo_size = vo_size
        self.use_query_bias = use_query_bias
        self.use_key_bias = use_key_bias
        self.use_value_bias = use_value_bias
        self.use_output_bias = use_output_bias
        self.use_qk_norm = use_qk_norm
        self.use_output_proj = use_output_proj

    @named_scope("eqx.nn.MultiheadAttention")
    def __call__(
        self,
        query: Float[Array, "q_seq q_size"],
        key_: Float[Array, "kv_seq k_size"],
        value: Float[Array, "kv_seq v_size"],
        bias: None | _Bias = None,
        mask: None | _Mask = None,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
        process_heads: None | _ProcessHeads = None,
    ) -> Float[Array, "q_seq o_size"]:
        """**Arguments:**

        - `query`: Query embedding. Should be a JAX array of shape
            `(query_seq_length, query_size)`.
        - `key_`: Key embedding. Should be a JAX array of shape
            `(kv_seq_length, key_size)`.
        - `value`: Value embedding. Should be a JAX array of shape
            `(kv_seq_length, value_size)`.
        - `bias` : Optional bias added to the attention logits. Should be a JAX array of
            shape `(query_seq_length, kv_seq_length)`. This is useful when implementing
            certain types of relative positional encodings.
        - `mask`: Optional mask preventing attention to certain positions. Should either
            be a JAX array of shape `(query_seq_length, kv_seq_length)`, or (for custom
            per-head masking) `(num_heads, query_seq_length, kv_seq_length)`. A value of
            `False` at a position indicates that position should be ignored.
        - `key`: A `jax.random.PRNGKey` used for dropout. Unused if `dropout = 0`.
            (Keyword only argument.)
        - `inference`: As [`equinox.nn.Dropout.__call__`][]. (Keyword only
            argument.)
        - `process_heads`: A function that takes in the query, key, and value heads and
            returns new query, key, and value heads. For example, this can be
            used to implement relative positional embeddings -
            see e.g. `RotaryPositionalEmbedding`for an example. (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(query_seq_length, output_size)`.
        """

        query_seq_length, _ = query.shape
        kv_seq_length, _ = key_.shape
        kv_seq_length2, _ = value.shape
        if kv_seq_length != kv_seq_length2:
            # query length can be different
            raise ValueError("key and value must both be sequences of equal length.")

        query_heads: Float[Array, "q_seq num_heads q_size"] = self._project(
            self.query_proj, query
        )
        key_heads: Float[Array, "kv_seq num_heads k_size"] = self._project(
            self.key_proj, key_
        )
        value_heads: Float[Array, "kv_seq num_heads v_size"] = self._project(
            self.value_proj, value
        )

        query_heads = self._normalize(self.query_norm, query_heads)
        key_heads = self._normalize(self.key_norm, key_heads)
        if process_heads is not None:
            q_shape, k_shape, v_shape = (
                query_heads.shape,
                key_heads.shape,
                value_heads.shape,
            )
            query_heads, key_heads, value_heads = process_heads(
                query_heads, key_heads, value_heads
            )

            if (
                query_heads.shape != q_shape
                or key_heads.shape != k_shape
                or value_heads.shape != v_shape
            ):
                raise ValueError(
                    "process_heads must not change the shape of the heads."
                )

        attn_fn = partial(
            dot_product_attention, dropout=self.dropout, inference=inference
        )
        keys = None if key is None else jax.random.split(key, query_heads.shape[1])

        if mask is not None and mask.ndim == 3:
            if bias is not None and bias.ndim == 3:
                # Batch `mask`, `bias` and `keys` down their 0-th dimension.
                attn = jax.vmap(attn_fn, in_axes=1, out_axes=1)(
                    query_heads, key_heads, value_heads, mask=mask, bias=bias, key=keys
                )
            else:
                # Batch `mask` and `keys` down their 0-th dimension.
                attn = jax.vmap(partial(attn_fn, bias=bias), in_axes=1, out_axes=1)(
                    query_heads, key_heads, value_heads, mask=mask, key=keys
                )
        else:
            if bias is not None and bias.ndim == 3:
                # Batch `bias` and `keys` down their 0-th dimension.
                attn = jax.vmap(partial(attn_fn, mask=mask), in_axes=1, out_axes=1)(
                    query_heads, key_heads, value_heads, bias=bias, key=keys
                )
            else:
                # Batch `keys` down its 0-th dimension.
                attn = jax.vmap(
                    partial(attn_fn, mask=mask, bias=bias), in_axes=1, out_axes=1
                )(query_heads, key_heads, value_heads, key=keys)
        attn = attn.reshape(query_seq_length, -1)

        return jax.vmap(self.output_proj)(attn)

    def _project(
        self, proj, x: Float[Array, "seq_length in_size"]
    ) -> Float[Array, "seq_length num_heads out_size"]:
        seq_length, _ = x.shape
        projection = jax.vmap(proj)(x)
        return projection.reshape(seq_length, self.num_heads, -1)

    def _normalize(
        self, norm, x: Float[Array, "seq_length num_heads qk_size"]
    ) -> Float[Array, "seq_length num_heads qk_size"]:
        if norm is None:
            return x
        else:
            x = rearrange(x, "len heads qk -> (len heads) qk")
            x = jax.vmap(norm)(x)
            return rearrange(x, "(len heads) qk -> len heads qk", heads=self.num_heads)
