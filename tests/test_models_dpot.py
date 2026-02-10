import jax
import jax.numpy as jnp
from context_flux_no.models.multiphysics.dpot import DPOTBlock, TimeAggregator
from einops import rearrange
from jaxtyping import Array, Float


def timeaggregator_original(
    x: Float[Array, "*grids time channels"],
    weights: Float[Array, "time channels channels"],
    gamma: Float[Array, " channels"],
    t: Float[Array, " time"],
) -> Float[Array, "*grids channels"]:
    """Forward pass of the TimeAggregator mirroring the original implementation at
    https://github.com/HaoZhongkai/DPOT/blob/main/models/dpot.py#L213-L234

    Note that the position of the grid axes is different from that of
    models.dpot.TimeAggregator.
    """
    gamma: Float[Array, "1 channels"] = jnp.expand_dims(gamma, 0)
    t: Float[Array, "time 1"] = jnp.expand_dims(t, -1)
    t_embed: Float[Array, "time channels"] = jnp.cos(t @ gamma)
    return jnp.einsum("tij,...ti->...j", weights, x * t_embed)


def test_timeaggregator():
    agg = TimeAggregator(10, 32, key=jax.random.key(0))

    x = jax.random.uniform(jax.random.key(0), (10, 32, 5, 5))
    x_orig = rearrange(x, "t c ... -> ... t c")

    out = agg(x)
    # Output shape
    assert out.shape == (32, 5, 5)

    # Output shape of the original TimeAggregator forward pass
    out_orig = timeaggregator_original(
        x_orig, agg.weights, agg.fourier_freqs, jnp.linspace(0, 1, agg.timesteps)
    )
    assert out_orig.shape == (5, 5, 32)

    # Identical computation performed
    assert jnp.all(jnp.isclose(rearrange(out, "c ... -> ... c"), out_orig))

    # 1D spatial data
    x_1d = jax.random.uniform(jax.random.key(0), (10, 32, 4))
    assert agg(x_1d).shape == (32, 4)

    # 3D spatial data
    x_3d = jax.random.uniform(jax.random.key(0), (10, 32, 7, 7, 7))
    assert agg(x_3d).shape == (32, 7, 7, 7)


def test_dpot_block():
    # 1D output shape
    block = DPOTBlock(
        num_spatial_dims=1,
        channels=32,
        max_frequency_modes=16,
        channels_hidden=32,
        key=jax.random.key(0),
    )
    img_1d = jnp.ones((32, 100))
    assert block(img_1d).shape == img_1d.shape

    # 2D output shape
    block = DPOTBlock(
        num_spatial_dims=2,
        channels=32,
        max_frequency_modes=16,
        channels_hidden=32,
        key=jax.random.key(0),
    )
    img_2d = jnp.ones((32, 100, 50))
    assert block(img_2d).shape == img_2d.shape

    # 3D output shape
    block = DPOTBlock(
        num_spatial_dims=3,
        channels=32,
        max_frequency_modes=16,
        channels_hidden=32,
        key=jax.random.key(0),
    )
    img_3d = jnp.ones((32, 100, 50, 40))
    assert block(img_3d).shape == img_3d.shape
