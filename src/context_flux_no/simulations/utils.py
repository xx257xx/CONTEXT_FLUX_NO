from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import xarray as xr
from jaxtyping import Array, PRNGKeyArray
from tqdm import tqdm


def sample_coefficients_uniform(
    key: PRNGKeyArray, coeff_range_dict: dict[str, tuple[float, float]]
) -> dict[str, float]:
    """
    Given a dictionary of coefficient names and a tuple indicating the range to sample
    from, return dictionary of coefficient names and values that are sampled from
    uniform distributions of the given range.
    """
    subkeys = jax.random.split(key, len(coeff_range_dict))
    return {
        name: float(
            jax.random.uniform(subkey, minval=coeff_range[0], maxval=coeff_range[1])
        )
        for subkey, (name, coeff_range) in zip(subkeys, coeff_range_dict.items())
    }


BASE_KEY_DICT = {"train": 0, "validation": 1, "test": 2}


def generate_dataset(
    n_coeffs: int,
    n_ics_per_coeff: int,
    pde_factory: Callable[[Any], eqx.Module],
    initial_condition_fn: Callable[[Array, PRNGKeyArray], Array],
    coeff_range_dict: dict,
    x_span: tuple[float, float],
    Nx: int,
    t_span: tuple[float, float],
    Nt: int,
    bc: Literal["periodic"] = "periodic",
    dataset_type: Literal["train", "validation", "test"] = "train",
    seed: int = 0,
):
    base_key = jax.random.key(BASE_KEY_DICT[dataset_type])
    keys = jax.random.split(jax.random.fold_in(base_key, seed), n_coeffs)

    solutions_all = []
    for key in tqdm(keys):
        key_coeff, *key_ics = jax.random.split(key, n_ics_per_coeff + 1)
        coeffs = sample_coefficients_uniform(key_coeff, coeff_range_dict)
        pde = pde_factory(**coeffs)

        solutions = []
        for key_ic in key_ics:
            sol = pde.solve(
                lambda u0: initial_condition_fn(u0, key_ic),
                x_span,
                Nx,
                t_span,
                Nt,
                bc=bc,
                verbose=False,
            )
            solutions.append(sol)

        solutions_all.append(xr.concat(solutions, "ic", data_vars="minimal"))
    return xr.concat(solutions_all, "pde")
