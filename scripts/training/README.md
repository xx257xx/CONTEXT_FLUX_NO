# Model training scripts

Model configuration and command line interface (CLI) is provided via the [Hydra](https://github.com/facebookresearch/hydra) library, with the hydra-submitit-plugin to interoperate with slurm.

To train multiphysics neural operator models (HyperFluxFNO, DPOT, DISCO), run the following script from the project top level directory:

(Running in the current terminal)
```bash
.venv/bin/python ./scripts/training/train_multiphysics.py model=hyperfluxfno
```
Note that the `--multirun` flag is need to submit the training as a slurm job.
(Submitting as a slurm job)
```bash
.venv/bin/python ./scripts/training/train_multiphysics.py model=hyperfluxfno --multirun
```
If submitting to a particular partition, or modifying any other sbatch arguments, use the hydra.launcher keyword:

(Submitting to a partition named h100)
```bash
.venv/bin/python ./scripts/training/train_multiphysics.py model=hyperfluxfno hydra.launcher.partition=h100 --multirun
```
Supported values for `model` are `hyperfluxfno`, `hyperfluxfno_local`, `dpot`, and `disco`.

Additional parameters can be passed to alter the model configuration. Alternatively, one can change the yaml files in scripts/training/configs to change default configuration values.

Once training starts, progress will be logged in wandb, corresponding to the wandb entity and project specified in /configs/config.yaml. 
Make sure the wandb entity value is appropriately set to the user's value.

The resulting checkpoint (corresponding to the best training loss) will be saved at /checkpoints directory.
