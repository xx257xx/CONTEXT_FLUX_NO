# Model training scripts

Model configuration and command line interface (CLI) is provided via the [Hydra](https://github.com/facebookresearch/hydra) library, with the hydra-submitit-plugin to interoperate with slurm.

To train the in-context Flux FNO model, run the following command in the main package folder:

```bash
.venv/bin/python ./scripts/training/train_incontext_fluxno.py --multirun
```

Note that the `--multirun` flag is need to submit the training as a slurm job. Additional parameters can be passed to alter the model configuration.

Once training starts, progress will be logged in wandb, corresponding to the wandb entity and project specified in /configs/config.yaml. 

The resulting checkpoint (corresponding to the best training loss) will be saved at /checkpoints directory.

If two training runs correspond to the same checkpoint directory, the training can randomly crash. Therefore it is good practice to move the checkpoints to an alternate directory after training finishes.