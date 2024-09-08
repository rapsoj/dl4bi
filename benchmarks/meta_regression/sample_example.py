#!/usr/bin/env python3
from pathlib import Path

import hydra
import wandb
from jax import random
from omegaconf import DictConfig, OmegaConf

from dsp.meta_regression.train_utils import (
    build_gp_dataloader,
    cfg_to_run_name,
    load_ckpt,
    sample_gif,
)


# NOTE: use the same configs as the Gaussian Process (GP) models
@hydra.main("configs/gp", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    path = f"results/gp/{cfg.data.name}/{cfg.kernel.kwargs.kernel.func}/{cfg.seed}/{run_name}"
    path = Path(path)
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=cfg.get("name", run_name),
        project=cfg.get("project", "Sampling"),
        reinit=True,  # allows reinitialization for multiple runs
    )
    cfg.data.batch_size = 1  # override GP batch argument
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    rng_data, rng_sample = random.split(rng)
    dataloader = build_gp_dataloader(cfg.data, cfg.kernel)
    state, _ = load_ckpt(path.with_suffix(".ckpt"))
    batch = next(dataloader(rng_data))
    paths = sample_gif(rng, state, batch)
    print(paths)


if __name__ == "__main__":
    main()
