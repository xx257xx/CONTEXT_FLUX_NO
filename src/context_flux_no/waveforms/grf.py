import abc
from collections.abc import Sequence
from functools import partial

import equinox as eqx
import gstools as gs
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float, PRNGKeyArray


class AbstractCovarianceFn(eqx.Module):
    """See https://www.cs.toronto.edu/~duvenaud/cookbook/ for information on different
    covariance functions"""

    @abc.abstractmethod
    def __call__(self, r: Float[Array, "*shape"]) -> Float[Array, "*shape"]: ...


class ExponentialCov(AbstractCovarianceFn):
    corr_length: float = eqx.field(static=True)
    marginal_std: float = eqx.field(static=True, default=1.0)

    def __call__(self, r: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
        return self.marginal_std**2 * jnp.exp(-jnp.abs(r) / self.corr_length)


class GaussianCov(AbstractCovarianceFn):
    corr_length: float = eqx.field(static=True)
    marginal_std: float = eqx.field(static=True, default=1.0)

    def __call__(self, r: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
        return self.marginal_std**2 * jnp.exp(-0.5 * (r / self.corr_length) ** 2)


class PeriodicCov(AbstractCovarianceFn):
    corr_length: float = eqx.field(static=True)
    marginal_std: float = eqx.field(static=True, default=1.0)
    period: float = eqx.field(static=True, default=1.0)

    def __call__(self, r: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
        return self.marginal_std * jnp.exp(
            -2 * (jnp.sin(jnp.pi * r / self.period) / self.corr_length) ** 2
        )


def mce_vector(x: Float[Array, " N"]) -> Float[Array, " 2*N-2"]:
    """Given a 1D vector x, representing the first row of a symmetric Toeplitz matrix X,
    compute the vector that gives the minimal circulant embedding (MCE) of X.

    That is, given y=mce_vector(x), Y=circ(y) gives the MCE of X.

    For more information, see Section 3.3 of [1].

    Reference
    [1] P. Taylor. Simulating Gaussian Random Fields and Solving Stochastic Differential
     Equations Using Bounded Wiener Increments. Doctoral Thesis, University of
     Manchester.
    """
    return jnp.concatenate((x, x[-2:0:-1]), axis=0)


def moce_vector(x: Float[Array, " N"]) -> Float[Array, " 2*N"]:
    """Given a 1D vector x, representing the first row of a symmetric Toeplitz matrix X,
    compute the vector that gives the minimal odd circulant embedding (MOCE) of X.

    That is, given y=moce_vector(x), Y=circ(y) gives the MOCE of X.

    For more information, see Section 3.3 of [1].

    Reference
    [1] P. Taylor. Simulating Gaussian Random Fields and Solving Stochastic Differential
     Equations Using Bounded Wiener Increments. Doctoral Thesis, University of
     Manchester.
    """
    return jnp.concatenate((x, x[::-1]), axis=0)


def assert_positive_real(
    x: Complex[Array, " *shape"], tol: float
) -> Float[Array, "* shape"]:
    """Checks if the given array is a positive real array upto some prespecified
    tolerance.

    If condition is not satisfied, raise an error. Otherwise, return the cleaned up
    array."""
    x_re, x_im = jnp.real(x), jnp.imag(x)
    cond = jnp.logical_and(jnp.min(x_re) < -tol, jnp.max(jnp.abs(x_im)) > tol)
    x = eqx.error_if(
        x, cond, "The given array does not have all positive real elements."
    )
    return jnp.clip(x_re, 0.0)


# TODO: rework the function interface to better avoid the concretization error
@partial(jax.jit, static_argnums=0, static_argnames="padding")
def generate_circulant_embedding_method_1d(
    Nx: int,
    dx: float,
    covariance: AbstractCovarianceFn,
    *,
    padding: int = 0,
    tol: float = 1e-9,
    key: PRNGKeyArray = jax.random.PRNGKey(0),
):
    """
    Generate a 1D isotropic stationary Gaussian random field with a prespecified
    covariance function using the circulant embedding method

    For more information see Section 4.2 of [1].
    All notations in the code is written to follow that of the above reference,
    except for the fact that the vector quantites are all now denoted in lower case.

    Nx: Number of grid points for the sampled random field
    dx: Spacing between the grid points
    tol: Tolerance for the violation of the positive real condition for the vector
        lambda_ (i.e., eigenvalues of the matrix B).
        One can try increasing the padding if this condition is violated.
        For float32 precision, tol=5e-6 is reasonable; for float64, tol=1e-15 is
        possible too.

    Note that if we are generating multiple samples, we can also use the imaginary part,
    which is also a valid sample for the Gaussian random field.

    Reference
    [1] P. Taylor. Simulating Gaussian Random Fields and Solving Stochastic Differential
     Equations Using Bounded Wiener Increments. Doctoral Thesis, University of
     Manchester.
    """
    a = covariance(jnp.arange(Nx + padding) * dx)
    b = mce_vector(a)
    M = len(b)
    lambda_ = jnp.sqrt(M) * jnp.fft.fft(b, norm="ortho")
    lambda_ = assert_positive_real(lambda_, tol)

    z_ = jax.random.normal(key, (2, M))
    z = z_[0] + 1j * z_[1]

    Lambda_half = jnp.diag(jnp.sqrt(lambda_))
    y_tilde = jnp.fft.ifft(Lambda_half @ z, norm="ortho")
    return jnp.real(y_tilde)[:Nx]


class GaussianRandomField1D(eqx.Module):
    covariance_fn: AbstractCovarianceFn

    def sample(self, x: Float[Array, " Nx"], key: PRNGKeyArray) -> Float[Array, "1 Nx"]:
        """Sample from the gaussian random field.

        Assume that the x coordinates are equispaced."""
        return jnp.expand_dims(
            generate_circulant_embedding_method_1d(
                len(x), x[1] - x[0], self.covariance_fn, key=key
            ),
            axis=0,
        )


class GaussianRandomField2D(eqx.Module):
    covariance_fn: gs.CovModel
    period: float
    num_modes: int
    _srf: gs.SRF

    def __init__(
        self, covariance_fn: gs.CovModel, period: float = 1.0, num_modes: int = 32
    ):
        self.covariance_fn = covariance_fn
        self.period = period
        self.num_modes = num_modes
        self._srf = gs.SRF(
            covariance_fn, generator="Fourier", period=period, mode_no=num_modes, seed=0
        )

    def sample(
        self, xs: tuple[Float[Array, " Nx"], Float[Array, " Ny"]], key: PRNGKeyArray
    ) -> Float[Array, "Nx Ny"]:
        """Sample from the gaussian random field.

        Assume that the x coordinates are equispaced."""
        with jax.default_device(jax.devices("cpu")[0]):
            seed = jax.random.bits(key)
        return self._srf(xs, seed=seed, mesh_type="structured")


class GaussianRandomField(eqx.Module):
    covariance_fns: tuple[gs.CovModel, ...]
    period: float
    num_modes: int
    dim: int
    _srfs: tuple[gs.SRF, ...]
    transforms: tuple

    def __init__(
        self,
        covariance_fns: Sequence[gs.CovModel],
        period: float = 1.0,
        num_modes: int = 32,
        transforms=None,
    ):
        dim = set(cov.dim for cov in covariance_fns)
        assert len(dim) == 1, "Covariance functions must all have the same dimension."

        self.covariance_fns = tuple(covariance_fns)
        self.dim = dim.pop()

        self.period = period
        self.num_modes = num_modes

        if transforms is None:
            normalizers = [None] * self.channels
        else:
            assert len(transforms) == self.channels, (
                "To use transforms, provide a transform per covariance function."
            )
            normalizers = transforms
        self.transforms = tuple(normalizers)

        self._srfs = tuple(
            [
                gs.SRF(
                    cov,
                    normalizer=norm,
                    generator="Fourier",
                    period=period,
                    mode_no=num_modes,
                )
                for cov, norm in zip(self.covariance_fns, self.transforms)
            ]
        )

    @property
    def channels(self) -> int:
        return len(self.covariance_fns)

    def sample(
        self, xs: tuple[Float[Array, " Nx"], Float[Array, " Ny"]], key: PRNGKeyArray
    ) -> Float[Array, "Nx Ny"]:
        """Sample from the gaussian random field.

        Assume that the x coordinates are equispaced."""
        with jax.default_device(jax.devices("cpu")[0]):
            keys = jax.random.split(key, self.channels)
            seeds = [jax.random.bits(k) for k in keys]
        fields = [
            srf(xs, seed=s, mesh_type="structured") for srf, s in zip(self._srfs, seeds)
        ]
        return np.stack(fields, axis=0)
