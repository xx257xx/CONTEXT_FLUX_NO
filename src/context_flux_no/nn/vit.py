import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .embedding import PatchEmbedding
from .transformer import TransformerEncoderBlock


class VisionTransformer(eqx.Module):
    patch_embedding: PatchEmbedding
    cls_token: Float[Array, "1 embedding_dim"]
    num_layers: int = eqx.field(static=True)

    def __init__(
        self,
        patch_size: tuple[int, int],
        in_channels: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, num_layers + 2)
        self.patch_embedding = PatchEmbedding(
            patch_size, in_channels, embedding_dim, key=keys[0]
        )
        self.cls_token = jax.random.truncated_normal(
            keys[1], lower=-2, upper=2, shape=(1, embedding_dim)
        )
        self.encoder_blocks = [
            TransformerEncoderBlock(
                num_heads, embedding_dim, hidden_dim, dropout, key=keys[i + 2]
            )
            for i in range(num_layers)
        ]

    def __call__(self, x: Float[Array, "channels height width"], *, key: PRNGKeyArray):
        ## TODO: Finish the implementation
        raise NotImplementedError

        x_embed: Float[Array, "num_patches embedding_dim"] = self.patch_embedding(x)
        # Add encoding

        x_embed = jnp.concatenate((self.cls_token, x_embed), axis=0)
        keys = jax.random.split(key, self.num_layers)
        for k, encoder_block in zip(keys, self.encoder_blocks):
            x_embed = encoder_block(x_embed, key=k)

        # Take either x_embed[0] or mean pool x_embed, then run it through an MLP to get the output
