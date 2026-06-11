import time

import grain.python as grain
import jax
from context_flux_no.data import TheWellDataSource


if __name__ == "__main__":
    jax.config.update("jax_default_device", jax.devices("gpu")[3])

    grain.config.update("py_debug_mode", True)

    source = TheWellDataSource(
        "./data/datasets",
        "euler_multi_quadrants_periodicBC",
        window_size=11,
        downsample_spatial=8,
        exclude_field_names=["pressure"],
    )
    ds = (
        grain.MapDataset.source(source)
        .seed(seed=0)
        .shuffle()
        .batch(batch_size=64)
        .to_iter_dataset()
    )
    it = iter(ds)

    for _ in range(2):
        t_start = time.time()
        batch = next(it)
        print(time.time() - t_start)
