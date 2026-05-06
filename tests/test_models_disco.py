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
from context_flux_no.models.multiphysics.disco.operatornet import (
    DownsamplingBlock,
    OperatorNetwork,
    UpSamplingBlock,
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


def test_downsampling_block():
    down_1d = DownsamplingBlock(
        num_spatial_dims=1,
        in_channels=4,
        out_channels=6,
        groups=6,
        padding_mode="REFLECT",
        groups_norm=6,
        hidden_channels=6,
        key=jax.random.key(0),
    )
    test_arr_1d = jnp.ones((4, 17))
    # Test 1D and odd spatial dim under jit
    assert eqx.filter_jit(down_1d)(test_arr_1d).shape == (6, 8)

    down_2d = DownsamplingBlock(
        num_spatial_dims=2,
        in_channels=3,
        out_channels=6,
        groups=6,
        padding_mode="REFLECT",
        groups_norm=6,
        hidden_channels=6,
        key=jax.random.key(0),
    )
    test_arr_2d = jnp.ones((3, 32, 32))
    # Test 2D and even spatial dim under jit
    assert eqx.filter_jit(down_2d)(test_arr_2d).shape == (6, 16, 16)

    down_3d = DownsamplingBlock(
        num_spatial_dims=3,
        in_channels=5,
        out_channels=6,
        groups=6,
        padding_mode="REFLECT",
        groups_norm=6,
        hidden_channels=6,
        key=jax.random.key(0),
    )
    test_arr_3d = jnp.ones((5, 9, 8, 10))
    # Test 3D and different dims per axis under jit
    assert eqx.filter_jit(down_3d)(test_arr_3d).shape == (6, 4, 4, 5)


def test_upsampling_block():
    up_1d = UpSamplingBlock(
        num_spatial_dims=1,
        in_channels=2,
        out_channels=6,
        groups=2,
        padding_mode="REFLECT",
        groups_norm=6,
        key=jax.random.key(0),
    )
    test_arrs_1d = (
        jnp.ones((2, 17)),
        jnp.ones((1, 34)),
    )
    # Test 1D under jit
    assert eqx.filter_jit(up_1d)(*test_arrs_1d).shape == (6, 34)

    up_2d = UpSamplingBlock(
        num_spatial_dims=2,
        in_channels=4,
        out_channels=6,
        groups=2,
        padding_mode="REFLECT",
        groups_norm=3,
        key=jax.random.key(0),
    )
    test_arrs_2d = (
        jnp.ones((4, 16, 16)),
        jnp.ones((2, 32, 32)),
    )
    # Test 2D under jit
    assert up_2d(*test_arrs_2d).shape == (6, 32, 32)

    up_3d = UpSamplingBlock(
        num_spatial_dims=3,
        in_channels=2,
        out_channels=6,
        groups=2,
        padding_mode="REFLECT",
        groups_norm=6,
        key=jax.random.key(0),
    )
    test_arrs_3d = (
        jnp.ones((2, 5, 7, 9)),
        jnp.ones((1, 10, 14, 18)),
    )
    # Test 3D and different dims per axis under jit
    assert up_3d(*test_arrs_3d).shape == (6, 10, 14, 18)


def test_operatornetwork():
    opnet_1d = OperatorNetwork(
        num_spatial_dims=1,
        channels=4,
        hidden_channels_base=8,
        groups_norm=4,
        boundary_condition="periodic",
        key=jax.random.key(0),
    )
    test_arr_1d = jnp.ones((4, 21))
    # Test 1D with odd spatial dimensions under jit
    assert eqx.filter_jit(opnet_1d)(test_arr_1d).shape == test_arr_1d.shape

    opnet_2d = OperatorNetwork(
        num_spatial_dims=2,
        channels=4,
        hidden_channels_base=8,
        groups_norm=4,
        boundary_condition="periodic",
        key=jax.random.key(0),
    )
    test_arr_2d = jnp.ones((4, 32, 32))
    # Test 2D with spatial dimensions that are multiples of 16 (the typical case)
    assert eqx.filter_jit(opnet_2d)(test_arr_2d).shape == test_arr_2d.shape

    opnet_3d = OperatorNetwork(
        num_spatial_dims=3,
        channels=4,
        hidden_channels_base=8,
        groups_norm=4,
        boundary_condition="periodic",
        key=jax.random.key(0),
    )
    test_arr_3d = jnp.ones((4, 21, 32, 18))
    # Test 3D with odd/multiple of 16/even but non-multiple of 16 dimensions
    eqx.filter_jit(opnet_3d)(test_arr_3d).shape == test_arr_3d.shape
