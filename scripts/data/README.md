# Data generation scripts

Model configuration and command line interface (CLI) is provided via the [Hydra](https://github.com/facebookresearch/hydra) library, with the hydra-submitit-plugin to interoperate with slurm.

## Cubic conservation law

Run the following command in the main package folder:

```bash
.venv/bin/python ./scripts/data/generate_cubic.py seed=0,1,2,3,4,5,6,7,8,9 --multirun
```