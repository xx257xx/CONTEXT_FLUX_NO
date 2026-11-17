"""Print the coefficient ranges and initial states of every 1D test set and its training
set, and the ratio of specific heats of the Euler sets (ICLR 2027 rebuttal plan, action
5h).

The shock-dominated shallow-water set of the submission was generated with ten times
the training interval of β. This script checks every other set for the same kind of
error: each line shows the range of every coefficient, and the spread and range of the
first frame, so a test set can be compared with the training set of its family.

Usage:

    uv run python scripts/data/check_coefficient_ranges.py --data-root ./data \
        --euler-root /path/to/euler/zarr/root

The 1D sets are the legacy HDF5 files under `datasets/` of `--data-root`. Every
dimension of `coeffs` other than `param` is reduced, so files that store the
coefficients as `(pde, param)` or as `(ic, param)` both work. The spread of the first
frame pools every channel, so for shallow water it mixes h and m. Without
`--euler-root` the Euler part is skipped. It reads the ratio of specific heats from
`scalars/gamma` of each zarr store (`data/sources.py:252-286`).
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import zarr


FAMILIES = ("cubic_no_source", "burgers_1d", "shallow_water_1d")


def describe_hdf5(path: Path) -> str:
    """Return the coefficient ranges and the spread and range of the first frame of one
    legacy HDF5 set, as one line.

    **Arguments:**

    - `path`: The HDF5 file.
    """
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        coeffs = dataset["coeffs"]
        other_dims = [dim for dim in coeffs.dims if dim != "param"]
        low, high = coeffs.min(other_dims), coeffs.max(other_dims)
        ranges = " ".join(
            f"{name}=[{float(low.sel(param=name)):+.2f},"
            f"{float(high.sel(param=name)):+.2f}]"
            for name in coeffs["param"].values
        )
        u0 = dataset["values"].isel(t=0)
        return (
            f"{ranges}  u0 std {float(u0.std()):.3f} "
            f"range [{float(u0.min()):+.2f},{float(u0.max()):+.2f}]"
        )


def describe_zarr(path: Path) -> str:
    """Return the range and number of the ratios of specific heats of one Euler store,
    as one line.

    **Arguments:**

    - `path`: The zarr store.
    """
    gamma = np.asarray(zarr.open_group(str(path), mode="r")["scalars/gamma"])
    return f"gamma=[{gamma.min():.3f},{gamma.max():.3f}] n={gamma.size}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-root", default="./data", help="Directory that holds datasets/."
    )
    parser.add_argument(
        "--euler-root", help="Directory searched for the zarr stores of 2D Euler."
    )
    arguments = parser.parse_args()

    datasets = Path(arguments.data_root) / "datasets"
    for family in FAMILIES:
        print(f"=== {family}")
        for path in sorted((datasets / family).rglob("*.hdf5")):
            print(f"  {str(path.relative_to(datasets)):70s} {describe_hdf5(path)}")

    if arguments.euler_root is not None:
        print("=== euler_2d")
        for path in sorted(Path(arguments.euler_root).rglob("*.zarr")):
            print(f"  {path.name:70s} {describe_zarr(path)}")


if __name__ == "__main__":
    main()
