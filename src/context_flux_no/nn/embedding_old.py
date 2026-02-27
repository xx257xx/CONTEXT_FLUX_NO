import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from equinox.nn._misc import named_scope
from jaxtyping import Array, Float, PRNGKeyArray


def unfold(
    x: Float[Array, "H W C"],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: list[tuple[int, int]] | str = "valid",
    *,
    flatten_patches: bool = True,
) -> Float[Array, "Ph Pw C*h*w"] | Float[Array, "Ph Pw C h w"]:
    """JAX version of torch.nn.functional.unfold, which extracts
    sliding local blocks from a given tensor

    If flatten_patches is True, each extracted patch will be a 1D tensor of shape
    (C*h*w); if flatten_patches is False, each patch
    will have shape (C, h, w).

    Here, h and w correspond to height and widths of the kernel."""

    patches = jax.lax.conv_general_dilated_patches(
        jnp.expand_dims(x, axis=0),
        kernel_size,
        stride,
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )[0]
    if not flatten_patches:
        h, w = kernel_size
        patches = rearrange(patches, "Ph Pw (C h w) -> Ph Pw C h w", h=h, w=w)
    return patches


class PatchEmbedding(eqx.Module):
    """
    Patch embedding layer for the vision transformer model, inspired by
    - Eqxvision PatchEmbed class:
        https://github.com/paganpasta/eqxvision/blob/main/eqxvision/layers/patch_embed.py#L11
    - Vision Transformer example in equinox:
        https://docs.kidger.site/equinox/examples/vision_transformer/
    """

    linear: eqx.nn.Linear
    patch_size: tuple[int, int] = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)
    flat_patch_positions: bool = eqx.field(static=True)

    def __init__(
        self,
        patch_size: tuple[int, int] | int,
        in_channels: int,
        embedding_dim: int,
        *,
        flat_patch_positions: bool = True,
        key: PRNGKeyArray,
    ):
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.linear = eqx.nn.Linear(
            in_features=in_channels * patch_size[0] * patch_size[1],
            out_features=embedding_dim,
            key=key,
        )
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.flat_patch_positions = flat_patch_positions

    def compute_padding(
        self, x: Float[Array, "height width in_channels"]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Compute the amount of padding required to make the image width/height
        integer multiples of patch width/height.

        The padding is applied at the right and bottom of the image"""
        img_height, img_width, _ = x.shape
        patch_height, patch_width = self.patch_size
        pad_h = img_height % patch_height
        pad_w = img_width % patch_width

        pad_h = 0 if pad_h == 0 else patch_height - pad_h
        pad_w = 0 if pad_w == 0 else patch_width - pad_w
        return ((0, pad_h), (0, pad_w))

    @named_scope("nn.PatchEmbedding")
    def __call__(
        self,
        x: Float[Array, "height width in_channels"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> (
        Float[Array, "num_patches embedding_dim"]
        | Float[Array, "num_patches_row num_patches_col embedding_dim"]
    ):
        # maybe use jax.ensure_compile_time_eval here?
        padding = self.compute_padding(x)

        flat_patches: Float[
            Array, "num_patches_h num_patches_w in_channels*patch_h*patch_w"
        ] = unfold(
            x,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=padding,
            flatten_patches=True,
        )
        if self.flat_patch_positions:
            flat_patches = rearrange(flat_patches, "Ph Pw N -> (Ph Pw) N")
            return jax.vmap(self.linear)(flat_patches)
        else:
            return jax.vmap(jax.vmap(self.linear))(flat_patches)
