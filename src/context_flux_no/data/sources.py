import abc
import contextlib
import os
from collections.abc import Sequence
from functools import cached_property
from itertools import accumulate
from pathlib import Path
from typing import Any, Iterator, Literal, Protocol

import fsspec
import grain
import h5py
import jax
import numpy as np
import yaml
import zarr
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

class HierarchicalStorage(Protocol):
    """Protocol for an object representing a hierarchical dataset.

    Examples are: h5py.File, zarr.Group."""

    attrs: Any

    def __getitem__(self, key: str, /) -> Any: ...

    def __contains__(self, key: str, /) -> bool: ...


def filter_strings(
    strings: Sequence[str],
    include_filters: list[str] | None = None,
    exclude_filters: list[str] | None = None,
) -> list[str]:
    # Logic from the original WellDataset code
    if include_filters is not None:
        retain_files = []
        for include_string in include_filters:
            retain_files += [f for f in strings if include_string in f]
        strings = retain_files
    if exclude_filters is not None:
        for exclude_string in exclude_filters:
            strings = [f for f in strings if exclude_string not in f]
    return strings


class WellDatasetSourceBase(grain.sources.RandomAccessDataSource):
    well_base_path: Path
    well_dataset_name: str
    well_split_name: Literal["train", "valid", "test"]
    include_filters: list[str] | None
    exclude_filters: list[str] | None
    filesystem: fsspec.AbstractFileSystem
    datapaths: list[str | Path]
    metadata_common: dict[str, Any]
    metadata_varying: dict[str, list[Any]]
    window_size: int
    exclude_field_names: tuple[str, ...]
    file_index_offsets: list[int]
    restrict_trajectory_lengths_to: int | None

    def __init__(
        self,
        well_base_path: Path | str,
        well_dataset_name: str,
        well_split_name: Literal["train", "valid", "test"] = "train",
        include_filters: list[str] | None = None,
        exclude_filters: list[str] | None = None,
        window_size: int = 21,
        use_normalization: bool = True,  # Only support z-score norm for now
        exclude_field_names: Sequence[str] = [],
        restrict_trajectory_lengths_to: int | None = None,
    ):
        self.well_base_path = Path(well_base_path)
        self.well_dataset_name = well_dataset_name
        self.well_split_name = well_split_name
        self.include_filters = include_filters
        self.exclude_filters = exclude_filters
        self.restrict_trajectory_lengths_to = restrict_trajectory_lengths_to

        dataset_dir = os.path.join(
            well_base_path, well_dataset_name, "data", well_split_name
        )
        self.filesystem = fsspec.url_to_fs(dataset_dir)[0]
        datapaths = self.get_datapaths()
        datapaths = filter_strings(
            datapaths, self.include_filters, self.exclude_filters
        )
        if len(datapaths) == 0:
            raise ValueError(f"""The given list does not contain any valid.hdf5
                        extension files.""")

        self.datapaths = datapaths

        self.metadata_common, self.metadata_varying = (
            self._check_consistency_and_build_metadata()
        )
        # Should implement getters and setters for self.window_size
        windows_per_trajectory = [
            n - window_size + 1
            if restrict_trajectory_lengths_to is None
            else restrict_trajectory_lengths_to - window_size + 1
            for n in self.metadata_varying["len_trajectories"]
        ]
        assert all(w > 0 for w in windows_per_trajectory), (
            """Given window_size is too large."""
        )
        self.window_size = window_size
        # TODO: make this into a cached property
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

        # If use_normalization is True, locate stats.yaml and load mean, std
        self.use_normalization = use_normalization
        if self.use_normalization:
            with open(
                self.well_base_path / self.well_dataset_name / "stats.yaml", "r"
            ) as f:
                stats = yaml.safe_load(f)
            for stat_name in ["mean", "std"]:
                self.metadata_common[stat_name] = np.concatenate(
                    [np.asarray(x).reshape(-1) for x in stats[stat_name].values()]
                )

    @abc.abstractmethod
    def get_datapaths(self) -> list[str | Path]: ...

    @contextlib.contextmanager
    @abc.abstractmethod
    def open_datapath(self, filepath: str | Path) -> Iterator[HierarchicalStorage]: ...

    def _check_consistency_and_build_metadata(self):
        """For the individual files in .hdf5, make sure that they have matching fields,
        shapes, etc. and return relevant metadata required for the __getitem__ logic.

        Corresponds to the _build_metadata() method of WellDataset."""

        metadata_common = {
            "dataset_name": set(),
            "n_spatial_dims": set(),
            "spatial_dims_shape": set(),
            "field_names": set(),
            "temporal_resolution": set(),
            "spatial_resolution": set(),
        }
        metadata_varying = {"n_trajectories": list(), "len_trajectories": list()}

        for datapath in self.datapaths:
            # Maybe make a light wrapper class around h5py.File to access relevant
            # information via properties and classmethods?

            with self.open_datapath(datapath) as file:
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

                t_grid = file["dimensions"]["time"]
                metadata_common["temporal_resolution"].add(t_grid[1] - t_grid[0])

                spat_res = []
                for dim_name in file["dimensions"].attrs["spatial_dims"]:
                    x_grid = file["dimensions"][dim_name]
                    spat_res.append(x_grid[1] - x_grid[0])
                metadata_common["spatial_resolution"].add(tuple(spat_res))

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

        with self.open_datapath(self.datapaths[file_idx]) as file:
            fields = []
            for rank, field_names in enumerate(self.valid_field_names):
                fields += [
                    file[f"t{rank}_fields"][n][
                        idx_traj, idx_window : idx_window + self.window_size
                    ]
                    for n in field_names
                ]
        item = pack(fields, self._pack_pattern)[0]
        # normalize if necessary
        if self.use_normalization:
            item = (item - self.metadata_common["mean"]) / self.metadata_common["std"]
        return rearrange(item, "t ... c -> t c ...")

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


class ZarrWellDatasetSource(WellDatasetSourceBase):
    def __init__(
        self,
        well_base_path: Path | str,
        well_dataset_name: str,
        well_split_name: Literal["train", "valid", "test"] = "train",
        include_filters: list[str] | None = None,
        exclude_filters: list[str] | None = None,
        window_size: int = 21,
        exclude_field_names: Sequence[str] = [],
        restrict_trajectory_lengths_to: int|None = None,
    ):
        super().__init__(
            well_base_path=well_base_path,
            well_dataset_name=well_dataset_name,
            well_split_name=well_split_name,
            include_filters=include_filters,
            exclude_filters=exclude_filters,
            window_size=window_size,
            use_normalization=False,  # TODO: compute stats for zarr datasets
            exclude_field_names=exclude_field_names,
            restrict_trajectory_lengths_to=restrict_trajectory_lengths_to,
        )

    def get_datapaths(self) -> list[str | Path]:  # TODO: fix type annotation
        dataset_dir = (
            self.well_base_path / self.well_dataset_name / "data" / self.well_split_name
        )
        dataset_dir = dataset_dir.resolve()
        datapaths = sorted(
            list(dataset_dir.glob("*.zarr")) + list(dataset_dir.glob("*.zarr.zip"))
        )
        return datapaths

    @contextlib.contextmanager
    def open_datapath(self, filepath: str | Path) -> Iterator[HierarchicalStorage]:
        root = zarr.open_group(filepath, mode="r")
        yield root  # TODO: fix protocol
        del root


class TheWellDataSource(WellDatasetSourceBase):
    # TODO: inherit from WellDatasetSourceBase and only implement the abstractmethods
    well_base_path: Path
    well_dataset_name: str
    well_split_name: Literal["train", "valid", "test", None]
    include_filters: list[str]
    exclude_filters: list[str]
    filesystem: fsspec.AbstractFileSystem
    datapaths: list[Path]
    metadata_common: dict[str, Any]
    metadata_varying: dict[str, list[Any]]
    window_size: int
    downsample_spatial: int
    exclude_field_names: tuple[str, ...]
    file_index_offsets: list[int]

    def __init__(
        self,
        well_base_path: Path | str,
        well_dataset_name: str,
        well_split_name: Literal["train", "valid", "test"] = "train",
        include_filters: list[str] = [],
        exclude_filters: list[str] = [],
        window_size: int = 21,
        downsample_spatial: int = 1,
        use_normalization: bool = True,  # Only support z-score norm for now
        exclude_field_names: Sequence[str] = [],
        restrict_trajectory_lengths_to: int|None = None,
    ):
        super().__init__(
                    well_base_path=well_base_path,
                    well_dataset_name=well_dataset_name,
                    well_split_name=well_split_name,
                    include_filters=include_filters,
                    exclude_filters=exclude_filters,
                    window_size=window_size,
                    use_normalization=use_normalization,
                    exclude_field_names=exclude_field_names,
                    restrict_trajectory_lengths_to=restrict_trajectory_lengths_to,
                )
        
        dataset_dir = os.path.join(
            well_base_path, well_dataset_name, "data", well_split_name
        )
        self.filesystem = fsspec.url_to_fs(dataset_dir)[0]
        datapaths = sorted(
            self.filesystem.glob(dataset_dir + "/*.h5")
            + self.filesystem.glob(dataset_dir + "/*.hdf5")
        )
        # Logic from the original WellDataset code
        if len(self.include_filters) > 0:
            retain_files = []
            for include_string in self.include_filters:
                retain_files += [f for f in datapaths if include_string in f]
            datapaths = retain_files
        if len(self.exclude_filters) > 0:
            for exclude_string in self.exclude_filters:
                datapaths = [f for f in datapaths if exclude_string not in f]
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
        # Should implement getters and setters for self.downsample_spatial
        self.metadata_common["spatial_resolution"] = tuple(
            i * downsample_spatial for i in self.metadata_common["spatial_resolution"]
        )
        self.downsample_spatial = downsample_spatial

        # If use_normalization is True, locate stats.yaml and load mean, std
        self.use_normalization = use_normalization
        with open(
            self.well_base_path / self.well_dataset_name / "stats.yaml", "r"
        ) as f:
            stats = yaml.safe_load(f)
        for stat_name in ["mean", "std"]:
            self.metadata_common[stat_name] = np.concatenate(
                [np.asarray(x).reshape(-1) for x in stats[stat_name].values()]
            )

    def get_datapaths(self) -> list[str | Path]:  # TODO: fix type annotation
        dataset_dir = os.path.join(
                    self.well_base_path, self.well_dataset_name, "data", eslf.well_split_name
                )
        datapaths = sorted(
                    self.filesystem.glob(dataset_dir + "/*.h5")
                    + self.filesystem.glob(dataset_dir + "/*.hdf5")
                )
        return datapaths

    @contextlib.contextmanager
    def open_datapath(self, filepath: str | Path) -> Iterator[HierarchicalStorage]:
        _f = self.filesystem.open(filepath, "rb", **IO_PARAMS["fsspec_params"])
        root = h5py.File(_f, "r", **IO_PARAMS["h5py_params"])
        yield root  # TODO: fix protocol
        root.close()
        _f.close()
    
