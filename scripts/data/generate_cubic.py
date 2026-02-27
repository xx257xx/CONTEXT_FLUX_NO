from pathlib import Path

import hydra
from context_flux_no.simulations.utils import generate_dataset
from omegaconf import DictConfig


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(cfg)
    dataset = generate_dataset(
        n_coeffs=cfg.n_coeffs,
        n_ics_per_coeff=cfg.n_ics_per_coeff,
        pde_factory=hydra.utils.get_method(cfg.pde_factory),
        initial_condition_fn=hydra.utils.instantiate(cfg.initial_condition).sample,
        coeff_range_dict=cfg.coeff_range_dict,
        x_span=tuple(cfg.x_span),
        Nx=cfg.Nx,
        t_span=tuple(cfg.t_span),
        Nt=cfg.Nt,
        dataset_type=cfg.dataset_type,
        seed=cfg.seed,
    )
    print("Dataset generated")
    savedir = Path(cfg.savepath)
    savedir.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(savedir / cfg.savename, engine="h5netcdf")


if __name__ == "__main__":
    main()
