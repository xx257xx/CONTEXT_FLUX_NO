import math

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, Int, PRNGKeyArray


class T5RelativePositionalEncoding(eqx.Module):
    """Re-implementation of the relative positional encoding used in the T5 paper [1].

    Implementation was largely inspired by those of [2, 3, 4].

    - num_encoding: Number of distinct encoding vectors used to bin relative positions
    - num_heads: Size of the positional encoding vectors.
    - max_distance: Maximum distance handled by the positional encoding. Larger
    distances will be truncated to this max_distance value.
    - bidirectional: Whether to encode both preceding (i<j) and succeeding distances
    (i>j) where i, j are indices of the query and key vectors. If False, the succeeding
    distances will be ignored.

    Note that this encoding scheme uses two different strategies - small distances are
    binned using their exact values; larger values are binned in a logarithmical manner,
    which corresponds to using larger bins for larger distances.

    [1] C. Raffel et al. Exploring the Limits of Transfer Learning with a Unified
    Text-to-Text Transformer. JMLR (2020).
    [2] https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    [3] https://github.com/AliHaiderAhmad001/T5-Relative-Position
    [4] https://gist.github.com/huchenxucs/c65524185e8e35c4bcfae4059f896c16
    """

    num_encodings: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    max_distance: int = eqx.field(static=True)
    bidirectional: bool = eqx.field(static=True)

    embedding: eqx.nn.Embedding

    def __init__(
        self,
        num_encodings: int = 32,
        num_heads=2,
        max_distance=128,
        bidirectional: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        self.embedding = eqx.nn.Embedding(
            num_encodings, num_heads, dtype=dtype, key=key
        )
        self.bidirectional = bidirectional
        self.num_encodings = num_encodings
        self.max_distance = max_distance
        self.num_heads = num_heads

    @property
    def effective_num_encodings(self) -> int:
        """Effective number of embeddings used. If not self.bidirectional, this is equal
        to self.num_encodings. Otherwise, it is half self.num_encodings to account for
        both positive and negative values."""
        return self.num_encodings // 2 if self.bidirectional else self.num_encodings

    def relative_position_to_embedding_bin(
        self, relative_position: Int[Array, "query_seq key_seq"]
    ) -> Int[Array, "query_seq key_seq"]:
        rel_pos = -relative_position  # Make preceding values positively distanced
        if self.bidirectional:
            offset = jnp.where(rel_pos < 0, self.effective_num_encodings, 0)
            rel_pos = jnp.abs(rel_pos)
        else:
            offset = jnp.zeros_like(rel_pos)
            rel_pos = jnp.clip(rel_pos, min=0)

        # Reserve half of the embeddings for exact values
        max_exact_binned_val = self.effective_num_encodings // 2

        # Use the other half of the embeddings to bin values logarithmically, up to
        # self.max_distance
        def bin_logarithmic(
            rel_pos_: Int[Array, "query_seq key_seq"],
        ) -> Int[Array, "query_seq key_seq"]:
            log_ratio = jnp.log(
                rel_pos_.astype(float) / max_exact_binned_val
            ) / math.log(self.max_distance / max_exact_binned_val)
            log_bin = max_exact_binned_val + log_ratio * (
                self.effective_num_encodings - max_exact_binned_val
            )
            return log_bin.astype(int)

        bin_value = jnp.where(
            rel_pos < max_exact_binned_val, rel_pos, bin_logarithmic(rel_pos)
        )
        bin_value = jnp.clip(bin_value, max=self.effective_num_encodings - 1)
        return offset + bin_value

    def __call__(
        self, query_seq: int, key_seq: int
    ) -> Float[Array, "num_heads query_seq key_seq"]:
        relative_position: Int[Array, "query_seq key_seq"] = jnp.arange(
            key_seq, dtype=int
        ) - jnp.arange(query_seq, dtype=int).reshape(-1, 1)
        """
                   k
             0   1   2   3
        q   -1   0   1   2
            -2  -1   0   1
            -3  -2  -1   0
        """
        embedding_bins: Int[Array, "query_seq key_seq"] = (
            self.relative_position_to_embedding_bin(relative_position)
        )
        pos_encs = jax.vmap(self.embedding)(rearrange(embedding_bins, "q k -> (q k)"))
        return rearrange(pos_encs, "(q k) h -> h q k", q=query_seq)


def rel_pos_to_bucket_bin(
    relative_position: Int[Array, "query_seq key_seq"],
    bidirectional=True,
    num_buckets=32,
    max_distance=128,
):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    Translate relative position to a bucket number for relative attention.
    The relative position is defined as memory_position - query_position, i.e.
    the distance in tokens from the attending position to the attended-to
    position.  If bidirectional=False, then positive relative positions are
    invalid.
    We use smaller buckets for small absolute relative_position and larger buckets
    for larger absolute relative_positions.  All relative positions >=max_distance
    map to the same bucket.  All relative positions <=-max_distance map to the
    same bucket.  This should allow for more graceful generalization to longer
    sequences than the model has been trained on.
    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer
    Returns:
        a Tensor with the same shape as relative_position, containing int32
        values in the range [0, num_buckets)
    """
    n = -relative_position
    if bidirectional:
        num_buckets //= 2
        offset = jnp.where(n < 0, num_buckets, 0)
        n = jnp.abs(n)
    else:
        n = jnp.clip(n, min=0)
        offset = 0
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    def bin_logarithmical(pos):
        log_ratio = jnp.log(pos / max_exact) / math.log(max_distance / max_exact)
        bin = max_exact + log_ratio * (num_buckets - max_exact)
        return bin.astype(int)

    bin_value = jnp.where(n < max_exact, n, bin_logarithmical(n))
    bin_value = jnp.clip(bin_value, max=num_buckets - 1)

    return offset + bin_value
