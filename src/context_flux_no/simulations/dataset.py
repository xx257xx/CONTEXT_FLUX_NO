from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import zarr

from context_flux_no.simulations.pde.base import AbstractHyperbolicConservationLaw


class ZarrWellDataset:
    savepath: Path
    root: zarr.Group
    dtype: Literal["float32", "float64"]

    # TODO: need read-only / writeable flag.
    def __init__(
        self,
        dataset_name: str,
        savepath: str | Path,
        n_trajectories: int = 1,
        dtype: Literal["float32", "float64"] = "float32",
        codec: zarr.abc.codec.Codec | None = None,
    ):
        self.savepath = Path(savepath)
        if self.savepath.exists():
            raise ValueError("A zarray dataset already exists at the save location.")

        self.root = zarr.group(self.savepath)
        self.root.attrs["dataset_name"] = dataset_name
        self.root.attrs["n_trajectories"] = n_trajectories
        self.dtype = dtype
        self.codec = codec

    @property
    def n_trajectories(self) -> int:
        return self.root.attrs["n_trajectories"]

    # TODO: probably, better to make this into a class method
    def initialize_with_pde(
        self,
        pde: AbstractHyperbolicConservationLaw,
        t_span: tuple[float, float],
        Nt: int,
        x_spans: Sequence[tuple[float, float]],
        Nxs: Sequence[int],
    ):
        self.root.attrs["grid_type"] = "cartesian"  # Other grid types not supported
        n_spatial_dims = pde.n_spatial_dims
        self.root.attrs["n_spatial_dims"] = n_spatial_dims

        dimensions = self.root.create_group("dimensions")
        # Works upto 3D
        spatial_dim_names = ["x", "y", "z"][:n_spatial_dims]
        dimensions.attrs["spatial_dims"] = spatial_dim_names
        time_arr = dimensions.create_array(
            name="time", shape=(Nt + 1,), dtype=self.dtype
        )
        time_arr[:] = np.linspace(*t_span, Nt+1, endpoint=True)
        time_arr.attrs["sample_varying"] = False

        for space_name, x_span, Nx in zip(spatial_dim_names, x_spans, Nxs):
            space_arr = dimensions.create_array(
                name=space_name, shape=(Nx,), dtype=self.dtype
            )
            x_grid = np.linspace(*x_span, Nx+1, endpoint=True)
            space_arr[:] = 0.5*(x_grid[:-1]+x_grid[1:]) # Cell centers
            space_arr.attrs["sample_varying"] = False
            space_arr.attrs["time_varying"] = False

        # Not creating the boundary_conditions group for now

        # Create empty dimension arrays
        scalars = self.root.create_group("scalars")
        scalar_names = list(pde.parameters.keys())
        scalars.attrs["field_names"] = scalar_names
        for s_name in scalar_names:
            scalar_arr = scalars.create_array(
                name=s_name, shape=(self.n_trajectories,), dtype=self.dtype
            )
            scalar_arr.attrs["sample_varying"] = True
            scalar_arr.attrs["time_varying"] = False

        field_name_dict = {r: [] for r in (0, 1, 2)}
        for r, f_name in pde.field_rank_names:
            field_name_dict[r].append(f_name)

        for r, f_names in field_name_dict.items():
            fields = self.root.create_group(f"t{r}_fields")
            fields.attrs["field_names"] = f_names

            shape = (self.n_trajectories, Nt + 1, *Nxs)
            if r != 0:
                shape = shape + (pde.n_spatial_dims**r,)
            for f_name in f_names:
                field_arr = fields.create_array(
                    name=f_name,
                    shape=shape,
                    dtype=self.dtype,
                    chunks=(1,) + shape[1:],
                    shards="auto",
                    compressor=self.codec,
                )
                field_arr.attrs["dim_varying"] = [True] * n_spatial_dims
                field_arr.attrs["sample_varying"] = True
                field_arr.attrs["time_varying"] = True

    def __len__(self) -> int:
        return self.n_trajectories

    def __getitem__(self, idx: int) -> dict[str, dict[str, np.ndarray]]:
        result_dict = {
            k: {} for k in ("scalars", "t0_fields", "t1_fields", "t2_fields")
        }
        for k in result_dict:
            grp = self.root.get_group(k)
            for f_name in grp.attrs["field_names"]:
                result_dict[k][f_name] = grp.get_array(f_name)[idx]
        return result_dict

    def __setitem__(self, idx: int, value: dict[str, dict[str, np.ndarray]]) -> None:
        for grp_name in ("scalars", "t0_fields", "t1_fields", "t2_fields"):
            grp = self.root.get_group(grp_name)
            for f_name, f_val in value[grp_name].items():
                grp[f_name][idx] = f_val

    def resize_dataset(self, n_trajectories_new: int):
        for grp_name in ("scalars", "t0_fields", "t1_fields", "t2_fields"):
            grp = self.root.get_group(grp_name)
            for f_name in grp.attrs["field_names"]:
                field_arr = grp.get_array(f_name)
                if field_arr.attrs["sample_varying"]:
                    field_arr.resize((n_trajectories_new,) + field_arr.shape[1:])
        self.root.attrs["n_trajectories"] = n_trajectories_new
