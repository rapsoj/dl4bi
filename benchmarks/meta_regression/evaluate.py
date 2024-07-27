#!/usr/bin/env python3
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from jax import random
from omegaconf import DictConfig

from dsp.meta_regression.train_utils import build_gp_dataloader, evaluate, load_ckpt


# NOTE: use the same configs as the Gaussian Process (GP) models
@hydra.main("configs/gp", version_base=None)
def main(cfg: DictConfig):
    d = HydraConfig.get().runtime.choices
    exp, kernel, model_cfg_name = d["exp"], d["kernel"], d["model"]
    rng = random.key(cfg.seed)
    rng_init, rng_data, rng_test = random.split(rng, 3)
    dataloader = build_gp_dataloader(cfg.exp, cfg.kernel)
    sample_batch = next(dataloader(rng_init))
    model_name = f"{exp}-{kernel}-{model_cfg_name}-seed-{cfg.seed}"
    model_ckpt_path = Path("results/gp") / (model_name + ".ckpt")
    model_state, _ = load_ckpt(model_ckpt_path, sample_batch)
    loss = evaluate(rng_test, model_state, dataloader, num_steps=5000)
    print(f"Loss: {loss:0.4f}")


if __name__ == "__main__":
    main()
