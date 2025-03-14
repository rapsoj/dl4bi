#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path
from typing import Mapping

import hydra
import jax
import jax.numpy as jnp
import optax
import wandb
from hydra.utils import instantiate
from jax import jit, random
from omegaconf import DictConfig, OmegaConf

from dl4bi.core.train import TrainState, cosine_annealing_lr, save_ckpt, train


@hydra.main("configs/simple", config_name="default", version_base=None)
def main(cfg: DictConfig):
    time = datetime.now().strftime("%Y-%m-%dT%H:%M")
    run_name = f"{cfg.model} - {time}"
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    dataloader = build_dataloader(cfg.data)
    lr_schedule = cosine_annealing_lr(
        cfg.train_num_steps,
        cfg.lr_peak,
        cfg.lr_pct_warmup,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.clip_max_norm),
        optax.yogi(lr_schedule),
    )
    model = instantiate(cfg.model)
    state = train(
        rng,
        model,
        optimizer,
        train_step,
        cfg.train_num_steps,
        dataloader,
        valid_step,
        cfg.valid_interval,
        cfg.valid_num_steps,
        dataloader,
    )
    path = f"results/{cfg.project}/{cfg.model}/{cfg.seed}/{run_name}"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloader(cfg: DictConfig):
    @jit
    def gen_batch(rng: jax.Array):
        rng_theta, rng_noise, rng = random.split(rng, 3)
        theta = random.uniform(
            rng_theta,
            (cfg.batch_size, 1),
            jnp.float32,
            cfg.theta.min,
            cfg.theta.max,
        )
        noise = cfg.noise * random.normal(rng_noise, theta.shape)
        return {"x": theta + noise, "theta": theta}

    def dataloader(rng: jax.Array):
        while True:
            rng_i, rng = random.split(rng)
            yield gen_batch(rng_i)

    return dataloader


@jit
def train_step(
    rng: jax.Array,
    state: TrainState,
    batch: Mapping,
    **kwargs,
):
    rng_dropout, rng_extra = random.split(rng)
    rngs = {"dropout": rng_dropout, "extra": rng_extra}
    x, theta = batch["x"], batch["theta"]

    def loss_fn(params):
        output = state.apply_fn({"params": params}, x, training=True, rngs=rngs)
        return output.nll(theta)

    nll, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), nll


@jit
def valid_step(
    rng: jax.Array,
    state: TrainState,
    batch: Mapping,
    **kwargs,
):
    x, theta = batch["x"], batch["theta"]
    rngs = {"extra": rng}
    output = state.apply_fn({"params": state.params}, x, training=False, rngs=rngs)
    return output.metrics(theta)


if __name__ == "__main__":
    main()
