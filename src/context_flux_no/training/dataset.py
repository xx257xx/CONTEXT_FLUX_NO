from dataclasses import replace
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

    u: Float[np.ndarray, "pde ic Nt dim Nx"]
    t: Float[np.ndarray, " Nt"]
    x: Float[np.ndarray, " Nx"]
    coeffs: Float[np.ndarray, " pde params"]
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

    @property
    def num_pde(self) -> int:
        """Number of distinct PDEs (different coefficient values) contained within the
        dataset."""
        return self.u.shape[0]

    @property
    def num_ic(self) -> int:
        """Number of distinct initial conditions per PDE contained in the dataset."""
        return self.u.shape[1]

    @property
    def dt(self) -> Float[Array, ""]:
        return self.t[1] - self.t[0]

    @property
    def dx(self) -> Float[Array, ""]:
        return self.x[1] - self.x[0]

    def split_by_time(self, time_idx: int) -> tuple[Self, Self]:
        """Split the dataset into two along the time axis at the given index value.
        Useful for creating training and extrapolation datasets."""
        u1, u2 = self.u[:, :, :time_idx], self.u[:, :, time_idx:]
        t1, t2 = self.t[:time_idx], self.t[time_idx:]
        return replace(self, u=u1, t=t1), replace(self, u=u2, t=t2)

    def downsample_time(self, downsample_factor: int) -> Self:
        u = self.u[:, :, ::downsample_factor]
        t = self.t[::downsample_factor]
        return replace(self, u=u, t=t)
