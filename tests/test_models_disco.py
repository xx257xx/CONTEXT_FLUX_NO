import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from context_flux_no.models.multiphysics.disco.attention_blocks import (
    AttentionBlock1D,
    SpatialAxialAttention,
    TemporalAttention,
)
from context_flux_no.models.multiphysics.disco.hypernet import (
    HyperNetwork,
    SpaceTimeBlock,
)


def test_attentionblock1d():
    atten = AttentionBlock1D(channels=384, num_heads=6, key=jax.random.key(0))

    test_arr1 = jax.random.normal(jax.random.key(1), shape=(10, 384))

    # Test 2D input
    assert atten(test_arr1, key=jax.random.key(2)).shape == test_arr1.shape
    # Test not passing in the random key
    assert atten(test_arr1).shape == test_arr1.shape

    test_arr2 = jax.random.normal(jax.random.key(3), shape=(5, 10, 384))

    # Test 3D input
    assert atten(test_arr2, key=jax.random.key(1)).shape == test_arr2.shape
    # Test vmap
    assert (
        eqx.filter_vmap(lambda x: atten(x, key=jax.random.key(1)))(test_arr2).shape
        == test_arr2.shape
    )
    # Test jit and vmap
    assert (
        eqx.filter_jit(eqx.filter_vmap(lambda x: atten(x, key=jax.random.key(1))))(
            test_arr2
        ).shape
        == test_arr2.shape
    )

    test_arr3 = jax.random.normal(jax.random.key(4), shape=(384,))
    # Test ValueError for 1D input
    with pytest.raises(ValueError):
        atten(test_arr3)

    test_arr4 = jax.random.normal(
        jax.random.key(seed=5),
        shape=(
            5,
            383,
        ),
    )
    # Test ValueError for wrong channel dimension
    with pytest.raises(ValueError):
        atten(test_arr4)


def test_temporal_attention():
    temporal = TemporalAttention(
        channels=384,
        num_heads=6,
        num_groups=6,
        layer_scale_init=1e-6,
        droppath=0.1,
        key=jax.random.key(0),
    )

    test_arr = jax.random.normal(jax.random.key(1), (10, 384, 10, 5, 3))
    # Test output shape
    assert temporal(test_arr, key=jax.random.key(0)).shape == test_arr.shape
    # Test jit
    assert (
        eqx.filter_jit(temporal)(test_arr, key=jax.random.key(0)).shape
        == test_arr.shape
    )
    # Test inference mode
    temporal_inf = eqx.nn.inference_mode(temporal)
    assert temporal_inf(test_arr).shape == test_arr.shape


def test_spatial_axial_attention():
    spatial = SpatialAxialAttention(
        num_spatial_dims=3,
        channels=384,
        num_heads=6,
        num_groups=12,
        layer_scale_init=1e-6,
        droppath=0.1,
        key=jax.random.key(0),
    )

    test_arr = jax.random.normal(jax.random.key(1), (4, 384, 10, 5, 3))

    # Test output shape
    assert spatial(test_arr[0], key=jax.random.key(2)).shape == test_arr.shape[1:]

    # Test jit
    assert (
        eqx.filter_jit(spatial)(test_arr[0], key=jax.random.key(2)).shape
        == test_arr.shape[1:]
    )

    # Test vmap
    assert (
        eqx.filter_vmap(lambda x: spatial(x, key=jax.random.key(2)))(test_arr).shape
        == test_arr.shape
    )


def test_spacetime_block():
    block = SpaceTimeBlock(
        channels=384, num_heads=6, droppath=0.1, key=jax.random.key(0)
    )
    test_arr = jnp.ones((5, 384, 2, 3, 4))

    # Test model under jit
    assert (
        eqx.filter_jit(block)(test_arr, key=jax.random.key(1)).shape == test_arr.shape
    )


def test_hypernetwork():
    hypernet = HyperNetwork(
        in_channels=2,
        patch_size=16,
        embedding_dim=384,
        num_blocks=4,
        droppath=0.1,
        num_heads=6,
        key=jax.random.key(0),
    )

    test_arr_1d = jnp.ones((5, 2, 64))
    test_arr_2d = jnp.ones((5, 2, 64, 32))
    test_arr_3d = jnp.ones((5, 2, 64, 32, 128))

    hypernet_jit = eqx.filter_jit(hypernet)
    # Test model for 1d input
    assert hypernet_jit(test_arr_1d, key=jax.random.key(1)).shape == (5, 384, 4, 1, 1)
    # Test model for 1d input
    assert hypernet_jit(test_arr_2d, key=jax.random.key(1)).shape == (5, 384, 4, 2, 1)
    # Test model for 1d input
    assert hypernet_jit(test_arr_3d, key=jax.random.key(1)).shape == (5, 384, 4, 2, 8)

    hypernet_inf = eqx.nn.inference_mode(hypernet)
    # Test inference mode
    assert hypernet_inf(test_arr_3d).shape == (5, 384, 4, 2, 8)
