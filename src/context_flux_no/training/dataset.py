from typing import Self

import equinox as eqx
import numpy as np
import xarray as xr
from jaxtyping import Array, Float


class PDEDataset(eqx.Module):
    """
    A class containing the PDE solutions and he corresponding coefficients.

    This serves as an intermediary between xr.Dataset and the rest of the JAX machinery
    since xarray by itself is not compatible with jax.jit, etc.
    """

    u: Float[np.ndarray, "samples Nt dim Nx"]
    t: Float[np.ndarray, " Nt"]
    x: Float[np.ndarray, " Nx"]
    coeffs: Float[np.ndarray, " samples params"]
    dim_names: list[str] = eqx.field(static=True)
    coeff_names: list[str] = eqx.field(static=True)

    @classmethod
    def from_xarray(cls, dataset: xr.Dataset) -> Self:
        u = dataset["values"].values
        t = dataset["t"].values
        x = dataset["x"].values
        coeffs = dataset["coeffs"].values
        dim_names = list(dataset["dim"].values)
        coeff_names = list(dataset["param"].values)
        return cls(u, t, x, coeffs, dim_names, coeff_names)

    def __len__(self) -> int:
        return self.u.shape[0]

    def __getitem__(self, idx) -> tuple[Array, Array]:
        return self.u[idx], self.coeffs[idx]
