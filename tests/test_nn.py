import jax
import jax.numpy as jnp
import pytest
from context_flux_no.nn.embedding import PatchEmbedding
from context_flux_no.nn.misc import apply_componentwise


def test_patchembedding_init():
    # Incompatible num_spatial_dims and patch_size
    with pytest.raises(ValueError):
        PatchEmbedding(1, (4, 3), 3, 32, key=jax.random.key(0))


def test_patchembedding_1d():
    patchembed_1d = PatchEmbedding(
        num_spatial_dims=1,
        patch_size=(4,),
        in_dim=3,
        embedding_dim=32,
        hidden_dim=8,
        num_hidden=1,
        key=jax.random.key(0),
    )
    # Output shape
    test_img = jnp.ones((3, 12))
    assert patchembed_1d(test_img).shape == (32, 3)

    # Proper padding when needed
    test_img2 = jnp.ones((3, 10))
    assert patchembed_1d.maybe_pad(test_img2).shape == (3, 12)

    # No padding when not needed
    assert patchembed_1d.maybe_pad(test_img).shape == (3, 12)


def test_patchembedding_2d():
    patchembed_2d = PatchEmbedding(
        num_spatial_dims=2,
        patch_size=(4, 3),
        in_dim=3,
        embedding_dim=32,
        hidden_dim=8,
        num_hidden=1,
        key=jax.random.key(0),
    )

    # Output shape
    test_img = jnp.ones((3, 12, 15))
    assert patchembed_2d(test_img).shape == (32, 3, 5)

    # Proper padding when needed
    test_img2 = jnp.ones((3, 10, 13))
    assert patchembed_2d.maybe_pad(test_img2).shape == (3, 12, 15)

    test_img3 = jnp.ones((3, 12, 13))
    assert patchembed_2d.maybe_pad(test_img3).shape == (3, 12, 15)

    test_img4 = jnp.ones((3, 10, 15))
    assert patchembed_2d.maybe_pad(test_img4).shape == (3, 12, 15)

    # No padding when not needed
    assert patchembed_2d.maybe_pad(test_img).shape == (3, 12, 15)


def test_patchembedding_3d():
    patchembed_3d = PatchEmbedding(
        num_spatial_dims=3,
        patch_size=(4, 3, 7),
        in_dim=3,
        embedding_dim=32,
        hidden_dim=8,
        num_hidden=1,
        key=jax.random.key(0),
    )

    # Output shape
    test_img = jnp.ones((3, 12, 15, 14))
    assert patchembed_3d(test_img).shape == (32, 3, 5, 2)

    # Proper padding when needed
    test_img2 = jnp.ones((3, 10, 13, 8))
    assert patchembed_3d.maybe_pad(test_img2).shape == (3, 12, 15, 14)

    test_img3 = jnp.ones((3, 12, 13, 8))
    assert patchembed_3d.maybe_pad(test_img3).shape == (3, 12, 15, 14)

    test_img4 = jnp.ones((3, 10, 15, 8))
    assert patchembed_3d.maybe_pad(test_img4).shape == (3, 12, 15, 14)

    test_img5 = jnp.ones((3, 10, 13, 14))
    assert patchembed_3d.maybe_pad(test_img5).shape == (3, 12, 15, 14)

    # No padding when not needed
    assert patchembed_3d.maybe_pad(test_img).shape == (3, 12, 15, 14)


def test_apply_componentwise():
    x_complex = jax.random.normal(jax.random.key(0), (100,), dtype=jnp.complex64)
    y_complex = jnp.sin(x_complex)
    y_complex_separate = apply_componentwise(jnp.sin)(x_complex)

    # Re[F(x)] is not equal to F(Re[x]) in JAX
    assert not jnp.array_equal(jnp.real(y_complex), jnp.sin(jnp.real(x_complex)))

    # Function separately applied to real and imaginary parts
    assert jnp.array_equal(jnp.real(y_complex_separate), jnp.sin(jnp.real(x_complex)))
    assert jnp.array_equal(jnp.imag(y_complex_separate), jnp.sin(jnp.imag(x_complex)))
