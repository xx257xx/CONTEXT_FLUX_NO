import hydra
import jax
from context_flux_no.simulations.utils import generate_dataset
from omegaconf import DictConfig


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(cfg)
    if cfg.gpu_id != "auto":
        jax.config.update("jax_default_device", jax.devices("gpu")[cfg.gpu_id])
    _ = generate_dataset(
        dataset_name=cfg.pde.name,
        n_coeffs=cfg.n_coeffs,
        n_ics_per_coeff=cfg.n_ics_per_coeff,
        pde_factory=hydra.utils.get_method(cfg.pde.pde_factory),
        initial_condition_fn=hydra.utils.instantiate(
            cfg.initial_condition.waveform
        ).sample,
        coeff_range_dict=cfg.pde.coeff_range_dict,
        x_span=tuple(cfg.x_span),
        Nx=cfg.Nx,
        t_span=tuple(cfg.t_span),
        Nt=cfg.Nt,
        dataset_type=cfg.dataset_type,
        seed=cfg.seed,
        savedir=cfg.savedir,
        filename=cfg.savename,
    )


if __name__ == "__main__":
    main()
