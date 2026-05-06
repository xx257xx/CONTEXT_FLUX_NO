# Data generation scripts

Model configuration and command line interface (CLI) is provided via the [Hydra](https://github.com/facebookresearch/hydra) library, with the hydra-submitit-plugin to interoperate with slurm.

## 1D Conservation law

### Flux functions
Supported values are `cubic_1d` for 1D cubic flux and `sine_1d` for 1D sine flux

### Initial conditions
Supported values are `grf` for Gaussian random field and `step` for step functions

## Example usage
To generate training data with default settings (Gaussian random field initial conditions and cubic flux) for multiple seeds using slurm job arrays (sbatch),
```bash
.venv/bin/python ./scripts/data/generate_cubic.py seed=0,1,2,3,4,5,6,7,8,9 --multirun
```

To generate test data for cubic flux and step initial conditions in the local terminal (recommand using tmux so that the process is not canceled when the terminal is exited),
```bash
.venv/bin/python ./scripts/data/generate_cubic.py initial_condition=step dataset_type=test seed=10
```

Similarly, for test data with sine flux and step initial conditions in the local terminal,
```bash
.venv/bin/python ./scripts/data/generate_cubic.py initial_condition=step pde=sine_1d dataset_type=test seed=12
```