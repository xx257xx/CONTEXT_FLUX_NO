"""Print how often the shallow-water data leaves the hyperbolic region (ICLR 2027
rebuttal plan, action 5i).

The flux of `simulations/pde/shallow_water.py:33-36` is
(γ m, α m²/h + β h²/2), so the squared wave speed is

    c² = α (α − γ) u² + γ β h,    u = m / h.

The Roe solver of the reference clips c² at `h_floor` (`shallow_water.py:84`), which
breaks the conservation of momentum where c² < 0. This happens only for α < γ. The
script evaluates c² on the stored cell states. These are a proxy for the Roe averages
that the solver clips: they include every saved frame, but not the solver substeps
between them.

Usage:

    uv run python scripts/data/check_shallow_water_hyperbolicity.py --data-root ./data

For every shallow-water file the script prints the fraction of trajectories with at
least one non-hyperbolic cell at any saved frame and at the first frame, the fraction
of all cells, and the fraction of trajectories with α < γ.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


# The number of trajectories read into memory at once.
CHUNK_SIZE = 2000
# The floor of h in the divisions of the reference solver (`shallow_water.py:65-66`).
H_FLOOR = 1e-8


def describe(path: Path) -> str:
    """Return the fractions of non-hyperbolic trajectories and cells of one file, as one
    line.

    **Arguments:**

    - `path`: A legacy HDF5 shallow-water set, with `values` of dimensions
        `(ic, t, dim, x)` in some order and `coeffs` of dimensions `(ic, param)`.
    """
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        values = dataset["values"].transpose("ic", "t", "dim", ...)
        coeffs = dataset["coeffs"]
        num_trajectories = dataset.sizes["ic"]
        any_frame = first_frame = negative_cells = num_cells = alpha_below = 0
        for start in range(0, num_trajectories, CHUNK_SIZE):
            chunk = slice(start, min(start + CHUNK_SIZE, num_trajectories))
            q = values.isel(ic=chunk).values
            alpha, beta, gamma = (
                coeffs.isel(ic=chunk).sel(param=name).values[:, None, None]
                for name in ("alpha", "beta", "gamma")
            )
            h, m = q[:, :, 0], q[:, :, 1]
            u = m / np.maximum(h, H_FLOOR)
            is_negative = alpha * (alpha - gamma) * u**2 + gamma * beta * h < 0
            any_frame += int(is_negative.any(axis=(1, 2)).sum())
            first_frame += int(is_negative[:, 0].any(axis=-1).sum())
            negative_cells += int(is_negative.sum())
            num_cells += is_negative.size
            alpha_below += int((alpha < gamma).sum())
    return (
        f"n={num_trajectories:6d}  any frame {any_frame / num_trajectories:6.1%}  "
        f"first frame {first_frame / num_trajectories:6.1%}  "
        f"cells {negative_cells / num_cells:8.3%}  "
        f"alpha<gamma {alpha_below / num_trajectories:6.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-root", default="./data", help="Directory that holds datasets/."
    )
    arguments = parser.parse_args()

    root = Path(arguments.data_root) / "datasets" / "shallow_water_1d"
    paths = sorted(root.rglob("*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No HDF5 file under {root}.")
    for path in paths:
        print(f"{str(path.relative_to(root)):65s} {describe(path)}")


if __name__ == "__main__":
    main()
