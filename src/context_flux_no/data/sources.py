import os
from collections.abc import Sequence
from functools import cached_property
from itertools import accumulate
from pathlib import Path
from typing import Any, Literal

import fsspec
import grain
import h5py
import jax
import numpy as np
from einops import pack, rearrange
from jaxtyping import Array, Float


# Taken from the_well: https://github.com/PolymathicAI/the_well/blob/master/the_well/data/utils.py#L33
IO_PARAMS = {
    "fsspec_params": {
        # "skip_instance_cache": True
        "cache_type": "blockcache",  # or "first" with enough space
        "block_size": 8 * 1024 * 1024,  # could be bigger
    },
    "h5py_params": {
        "driver_kwds": {  # only recent versions of xarray and h5netcdf allow this correctly
            "page_buf_size": 8 * 1024 * 1024,  # this one only works in repacked files
            "rdcc_nbytes": 8 * 1024 * 1024,  # this one is to read the chunks
        }
    },
}


class TheWellDataSource(grain.sources.RandomAccessDataSource):
    well_base_path: Path | str
    well_dataset_name: str
    well_split_name: Literal["train", "valid", "test", None]
    filesystem: fsspec.AbstractFileSystem
    datapaths: list[Path]
    metadata_common: dict[str, Any]
    metadata_varying: dict[str, list[Any]]
    window_size: int
    exclude_field_names: tuple[str, ...]
    file_index_offsets: list[int]

    def __init__(
        self,
        well_base_path: Path | str,
        well_dataset_name: str,
        well_split_name: Literal["train", "valid", "test"] = "train",
        window_size: int = 21,
        exclude_field_names: Sequence[str] = [],
    ):
        dataset_dir = os.path.join(
            well_base_path, well_dataset_name, "data", well_split_name
        )
        self.filesystem = fsspec.url_to_fs(dataset_dir)[0]
        datapaths = sorted(
            self.filesystem.glob(dataset_dir + "/*.h5")
            + self.filesystem.glob(dataset_dir + "/*.hdf5")
        )

        if len(datapaths) == 0:
            raise ValueError(f"""The directory {dataset_dir} does not contain any .hdf5
             extension files.""")

        self.datapaths = datapaths

        self.metadata_common, self.metadata_varying = (
            self._check_consistency_and_build_metadata()
        )
        # Should implement getters and setters for self.window_size
        windows_per_trajectory = [
            n - window_size + 1 for n in self.metadata_varying["len_trajectories"]
        ]
        assert all(w > 0 for w in windows_per_trajectory), (
            """Given window_size is too large."""
        )
        self.window_size = window_size
        self.file_index_offsets = list(
            accumulate(
                (
                    n_traj * n_win
                    for (n_traj, n_win) in zip(
                        self.metadata_varying["n_trajectories"], windows_per_trajectory
                    )
                ),
                initial=0,
            )
        )
        self.exclude_field_names = tuple(exclude_field_names)

    def _check_consistency_and_build_metadata(self):
        """For the individual files in .hdf5, make sure that they have matching fields,
        shapes, etc. and return relevant metadata required for the __getitem__ logic.

        Corresponds to the _build_metadata() method of WellDataset."""

        metadata_common = {
            "dataset_name": set(),
            "n_spatial_dims": set(),
            "spatial_dims_shape": set(),
            "field_names": set(),
        }
        metadata_varying = {"n_trajectories": list(), "len_trajectories": list()}

        for datapath in self.datapaths:
            # Maybe make a light wrapper class around h5py.File to access relevant
            # information via properties and classmethods?

            with (
                self.filesystem.open(
                    datapath, "rb", **IO_PARAMS["fsspec_params"]
                ) as _f,
                h5py.File(_f, "r", **IO_PARAMS["h5py_params"]) as file,
            ):
                # Query common metadata and assert they are unique
                for k in ("dataset_name", "n_spatial_dims"):
                    metadata_common[k].add(file.attrs[k])
                metadata_common["spatial_dims_shape"].add(
                    tuple(
                        [
                            file["dimensions"][d].shape[-1]
                            for d in file["dimensions"].attrs["spatial_dims"]
                        ]
                    )
                )
                # Check the time varying attribute?
                metadata_common["field_names"].add(
                    tuple([tuple(file[f"t{j}_fields"].keys()) for j in range(3)])
                )

                for metadata_name, val in metadata_common.items():
                    assert (
                        len(val) == 1
                    ), f"""Multiple values of {metadata_name} found in specified path.
                        """

                # Query varying metadata
                metadata_varying["n_trajectories"].append(
                    int(file.attrs["n_trajectories"])
                )
                metadata_varying["len_trajectories"].append(
                    file["dimensions"]["time"].shape[-1]
                )
        metadata_common = jax.tree.map(lambda _set: _set.pop(), metadata_common)
        return metadata_common, metadata_varying

    def __len__(self) -> int:
        return self.file_index_offsets[-1]

    def __getitem__(self, idx: int) -> Float[Array, "time *spatial_dims channel"]:  # ty: ignore[invalid-method-override]
        file_idx = int(np.searchsorted(self.file_index_offsets, idx, side="right")) - 1
        idx_local = idx - self.file_index_offsets[file_idx]
        idx_window, idx_traj = divmod(
            idx_local, self.metadata_varying["n_trajectories"][file_idx]
        )

        with (
            self.filesystem.open(
                self.datapaths[file_idx], "rb", **IO_PARAMS["fsspec_params"]
            ) as _f,
            h5py.File(_f, "r", **IO_PARAMS["h5py_params"]) as file,
        ):
            fields = []
            for rank, field_names in enumerate(self.valid_field_names):
                fields += [
                    file[f"t{rank}_fields"][n][
                        idx_traj, idx_window : idx_window + self.window_size
                    ]
                    for n in field_names
                ]

        return rearrange(pack(fields, self._pack_pattern)[0], "t ... c -> t c ...")

    @cached_property
    def valid_field_names(self) -> tuple[tuple[str, ...], ...]:
        valid_names = []
        for names in self.metadata_common["field_names"]:
            valid_names.append(
                tuple(n for n in names if n not in self.exclude_field_names)
            )
        return tuple(valid_names)

    @cached_property
    def _pack_pattern(self) -> str:
        return " ".join(
            [
                "t",
                *[f"x{i}" for i in range(self.metadata_common["n_spatial_dims"])],
                "*",
            ]
        )
