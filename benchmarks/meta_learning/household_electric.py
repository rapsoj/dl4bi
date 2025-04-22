#!/usr/bin/env python3
from pathlib import Path

import hydra
import jax
import pandas as pd
import wandb
from hydra.utils import instantiate
from jax import random
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler

from dl4bi.core.train import (
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.tabular import TabularData
from dl4bi.meta_learning.utils import cfg_to_run_name


@hydra.main("configs/household_electric", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    rng_data, rng_train, rng_test = random.split(rng, 3)
    train_dataloader, valid_dataloader, test_dataloader = build_dataloaders(rng_data)
    optimizer = instantiate(cfg.optimizer)
    model = instantiate(cfg.model)
    state = train(
        rng_train,
        model,
        optimizer,
        model.train_step,
        cfg.train_num_steps,
        train_dataloader,
        model.valid_step,
        cfg.valid_interval,
        cfg.valid_num_steps,
        valid_dataloader,
    )
    metrics = evaluate(
        rng_test,
        state,
        model.valid_step,
        valid_dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloaders(
    rng: jax.Array,
    batch_size: int = 16,
    num_ctx_min: int = 16,
    num_ctx_max: int = 128,
    num_test: int = 256,
):
    def build_dataloader(y, X):
        def dataloader(rng: jax.Array):
            while True:
                rng_i, rng = random.split(rng)
                yield TabularData(x=X, f=y).batch(
                    rng_i,
                    num_ctx_min,
                    num_ctx_max,
                    num_test,
                    obs_noise=None,
                    test_includes_ctx=False,
                    batch_size=batch_size,
                )

        return dataloader

    (y_train, X_train), (y_valid, X_valid), (y_test, X_test) = load_data(rng)
    return (
        build_dataloader(y_train, X_train),
        build_dataloader(y_valid, X_valid),
        build_dataloader(y_test, X_test),
    )


def load_data(rng: jax.Array):
    # source: https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set
    df = pd.read_csv("cache/household_power_consumption.txt", sep=";").dropna()
    df["dt"] = pd.to_datetime(df.Date + " " + df.Time, dayfirst=True)
    df["year"] = df.dt.dt.year
    df["month"] = df.dt.dt.month
    df["day"] = df.dt.dt.day
    df = df.drop(columns=["Date", "Time", "dt"])
    yX = df.values
    N = yX.shape[0]
    pct_train, pct_test = 0.8, 0.1
    num_train, num_test = int(pct_train * N), int(pct_test * N)
    perm = random.permutation(rng, N)
    yX = yX[perm]
    yX_train, yX_valid, yX_test = (
        yX[:num_train],
        yX[num_train:-num_test],
        yX[-num_test:],
    )
    scaler = StandardScaler()
    yX_train = scaler.fit_transform(yX_train)
    yX_valid = scaler.transform(yX_valid)
    yX_test = scaler.transform(yX_test)
    y_train, X_train = yX_train[:, [0]], yX_train[:, 1:]
    y_valid, X_valid = yX_valid[:, [0]], yX_valid[:, 1:]
    y_test, X_test = yX_test[:, [0]], yX_test[:, 1:]
    return (y_train, X_train), (y_valid, X_valid), (y_test, X_test)


if __name__ == "__main__":
    main()
