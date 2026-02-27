from functools import partial

import equinox as eqx
import jax
from equinox.nn._misc import named_scope
from jaxtyping import Array, Float, PRNGKeyArray


class TransformerEncoderBlock(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    layernorms: tuple[eqx.nn.LayerNorm, eqx.nn.LayerNorm]
    mlp: eqx.nn.Sequential
    num_heads: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    dropout: float = eqx.field(static=True)

    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        *,
        key: PRNGKeyArray,
    ):
        key_a, key_l1, key_l2 = jax.random.split(key, 3)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=embedding_dim,
            dropout_p=dropout,
            key=key_a,
        )
        self.layernorms = (
            eqx.nn.LayerNorm(embedding_dim),
            eqx.nn.LayerNorm(embedding_dim),
        )
        self.mlp = eqx.nn.Sequential(
            [
                eqx.nn.Linear(embedding_dim, hidden_dim, key=key_l1),
                eqx.nn.Lambda(jax.nn.gelu),
                eqx.nn.Dropout(p=dropout),
                eqx.nn.Linear(hidden_dim, embedding_dim, key=key_l2),
                eqx.nn.Dropout(p=dropout),
            ]
        )

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    @named_scope("nn.TransformerEncoderBlock")
    def __call__(
        self,
        x: Float[Array, "sequence_length embedding_dim"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "sequence_length embedding_dim"]:
        # RNG keys for dropout
        key1, key2 = jax.random.split(key, 2)

        # vmap to not normalize over the first dimension
        x_norm = jax.vmap(self.layernorms[0])(x)
        x = x + self.attention(x_norm, x_norm, x_norm, key=key1)
        x_norm = jax.vmap(self.layernorms[1])(x)
        # vmap as mlp also expects vectors otherwise
        return x + jax.vmap(partial(self.mlp, key=key2))(x_norm)
