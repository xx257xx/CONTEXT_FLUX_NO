import abc
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, PRNGKeyArray


class AbstractCovarianceFn(eqx.Module):
    @abc.abstractmethod
    def __call__(self, r: Float[Array, "*shape"]) -> Float[Array, "*shape"]: ...


class ExponentialCov(AbstractCovarianceFn):
    corr_length: float = eqx.field(static=True)
    marginal_std: float = eqx.field(static=True, default=1.0)

    def __call__(self, r: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
        return self.marginal_std * jnp.exp(-jnp.abs(r) / self.corr_length)


class GaussianCov(AbstractCovarianceFn):
    corr_length: float = eqx.field(static=True)
    marginal_std: float = eqx.field(static=True, default=1.0)

    def __call__(self, r: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
        return self.marginal_std * jnp.exp(-((r / self.corr_length) ** 2))


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
