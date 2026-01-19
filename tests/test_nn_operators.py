import jax
import jax.numpy as jnp
from context_flux_no.nn.operators.adaptivefourier import AdaptiveFourier
from context_flux_no.nn.operators.fourier_utils import valid_frequency_inds


def test_valid_frequency_inds():
    freq_1d = jnp.fft.rfftfreq(10)

    # Zero mode
    inds = valid_frequency_inds(freq_1d.shape, (0,))
    inds_expected = (
        jnp.asarray(
            [
                0,
            ],
            dtype=inds[0].dtype,
        ),
    )
    assert all(jax.tree.map(jnp.array_equal, inds, inds_expected))

    # Less than Nyquist freq
    inds = valid_frequency_inds(freq_1d.shape, (3,))
    inds_expected = (
        jnp.asarray(
            [0, 1, 2, 3],
            dtype=inds[0].dtype,
        ),
    )
    assert all(jax.tree.map(jnp.array_equal, inds, inds_expected))

    # More than Nyquist freq
    inds = valid_frequency_inds(freq_1d.shape, (8,))
    inds_expected = (
        jnp.asarray(
            [0, 1, 2, 3, 4, 5],
            dtype=inds[0].dtype,
        ),
    )
    assert all(jax.tree.map(jnp.array_equal, inds, inds_expected))

    # 2D case
    freqs_2d = jnp.meshgrid(jnp.fft.fftfreq(13), jnp.fft.rfftfreq(10), indexing="ij")[0]
    inds = valid_frequency_inds(freqs_2d.shape, (1, 3))
    inds_expected = (
        jnp.asarray(
            [0, 1, 12],
            dtype=inds[0].dtype,
        ).reshape(-1, 1),
        jnp.asarray([0, 1, 2, 3], dtype=inds[0].dtype).reshape(1, -1),
    )
    assert all(jax.tree.map(jnp.array_equal, inds, inds_expected))


def test_adaptive_fourier():
    # 1D output shape
    afno1d = AdaptiveFourier(
        num_spatial_dims=1,
        in_channels=32,
        out_channels=32,
        max_frequency_modes=(10,),
        hidden_channels=64,
        num_blocks=8,
        key=jax.random.key(0),
    )
    img_1d = jnp.ones((32, 100))
    assert afno1d(img_1d).shape == img_1d.shape

    # 2D output shape
    afno2d = AdaptiveFourier(
        num_spatial_dims=2,
        in_channels=32,
        out_channels=32,
        max_frequency_modes=(10, 7),
        hidden_channels=64,
        num_blocks=8,
        key=jax.random.key(0),
    )
    img_2d = jnp.ones((32, 100, 100))
    assert afno2d(img_2d).shape == img_2d.shape

    # 3D output shape
    afno3d = afno2d = AdaptiveFourier(
        num_spatial_dims=3,
        in_channels=32,
        out_channels=32,
        max_frequency_modes=(10, 7, 8),
        hidden_channels=64,
        num_blocks=8,
        key=jax.random.key(0),
    )
    img_3d = jnp.ones((32, 100, 100, 100))
    assert afno3d(img_3d).shape == img_3d.shape
