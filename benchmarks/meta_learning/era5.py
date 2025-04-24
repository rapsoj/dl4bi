#!/usr/bin/env python3
import os
from datetime import timedelta
from pathlib import Path

import cdsapi
import hydra
import jax
import numpy as np
import pandas as pd
import wandb
import xarray as xr
from hydra.utils import instantiate
from jax import random
from omegaconf import DictConfig, OmegaConf

from dl4bi.core.train import (
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.spatiotemporal import SpatiotemporalData
from dl4bi.meta_learning.utils import cfg_to_run_name


@hydra.main("configs/era5", config_name="default", version_base=None)
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


# TODO(danj): update
def build_dataloaders(
    batch_size: int = 16,
    num_ctx_min: int = 56,  # ~5% of 1,125
    num_ctx_max: int = 287,  # ~25.5% of 1,125
    num_test: int = 225,  # 287 + 225 = 512 = L
    test_dataset: str = "western_europe",  # western_europe, northern_europe
):
    grid_res = 0.5
    B, H_deg, W_deg, T_hrs, T_hrs_delta = batch_size, 7.5, 7.5, 30, 6
    H, W = int(H_deg / grid_res), int(W_deg / grid_res)
    T = T_hrs // T_hrs_delta
    df_train = load_data("central_europe")
    df_test = load_data(test_dataset)
    t_min = df_train.valid_time.min()
    df_train["hour"] = (df_train.valid_time - t_min) / pd.Timedelta(hours=1)
    hour_mu, hour_std = df_train.hour.mean(), df_train.hour.std()
    lat_mu, lat_std = df_train.latitude.mean(), df_train.latitude.std()
    lng_mu, lng_std = df_train.longitude.mean(), df_train.longitude.std()
    elev_mu, elev_std = df_train.z.mean(), df_train.z.std()
    temp_mu, temp_std = df_train.t2m.mean(), df_train.t2m.std()

    def standardize(df: pd.DataFrame):
        df["hour"] = (df.valid_time - t_min) / pd.Timedelta(hours=1)
        df["hour_std"] = (df.hour - hour_mu) / hour_std
        df["lat_std"] = (df.latitude - lat_mu) / lat_std
        df["lng_std"] = (df.longitude - lng_mu) / lng_std
        df["elev_std"] = (df.z - elev_mu) / elev_std
        df["temp_std"] = (df.t2m - temp_mu) / temp_std
        df.rename(columns={"t2m": "temp"}, inplace=True)
        df = df.sort_values(
            by=["hour_std", "lat_std", "lng_std"], ascending=[True, False, True]
        )
        return df.drop(columns=["z", "hour"]).reset_index(drop=True)

    df_train_std = standardize(df_train)
    df_test_std = standardize(df_test)
    # TODO(danj): sample hourly, reshape
    # 0. sort by time, lat, lng
    # 1. resample original data every 6 hours
    # 2. reshape into (T, H, W, D)
    # 3. reduce time from (T, H, W, 1) to (T,)
    # 4. use SpatiotemporalData

    def build_dataloader(df: pd.DataFrame, is_callback: bool = False):
        time_uniq = df.valid_time.unique()
        time_choices = time_uniq[time_uniq <= time_uniq.max() - timedelta(hours=T)]
        lat_uniq, lng_uniq = df.latitude.unique(), df.longitude.unique()
        lat_choices = lat_uniq[lat_uniq <= lat_uniq.max() - H_deg]
        lng_choices = lng_uniq[lng_uniq <= lng_uniq.max() - W_deg]
        time_diffs = np.array([timedelta(hours=T_hrs_delta * i) for i in range(T)])
        data_cols = ["hour_std", "lat_std", "lng_std", "elev_std", "temp_std"]

        def dataloader(rng: jax.Array):
            while True:
                rng_t, rng_lat, rng_lng, rng_b, rng = random.split(rng, 5)
                time_starts_idx = random.choice(rng_t, len(time_choices), (B,))
                time_starts = time_choices[time_starts_idx]
                lat_starts = random.choice(rng_lat, lat_choices, (B,))
                lng_starts = random.choice(rng_lng, lng_choices, (B,))
                starts = zip(time_starts, lat_starts, lng_starts)
                for time_start, lat_start, lng_start in starts:
                    ts = time_start + time_diffs
                    dft = df[df.valid_time.isin(ts)]
                    dft = dft[
                        (dft.latitude >= lat_start) & (dft.latitude < lat_start + H_deg)
                    ]
                    dft = dft[
                        (dft.longitude >= lng_start)
                        & (dft.longitude < lng_start + W_deg)
                    ]
                    data = dft[data_cols].values.reshape(T, H, W, -1)
                    t = data[..., [0]]
                    s = data[..., 1:3]
                    x = data[..., [3]]
                    f = data[..., [-1]]

        return dataloader

    return (
        build_dataloader(df_train_std),
        build_dataloader(df_train_std),
        build_dataloader(df_test_std),
    )


def load_data(region: str = "central_europe"):
    download_if_not_cached()
    ds = xr.open_mfdataset(f"cache/era5/{region}/2019_*.nc", combine="by_coords")
    df = ds.to_dataframe()[["t2m", "z"]].reset_index()
    return df


def download_if_not_cached():
    client = None
    variables = ["2m_temperature", "geopotential"]
    times = [f"{t:02d}:00" for t in range(24)]
    central_europe = [53, 8, 42, 28]  # N, W, S, E
    western_europe = [53, -4, 42, 8]
    northern_europe = [62, 8, 53, 28]
    grid = [0.5, 0.5]

    for region, area in [
        ("central_europe", central_europe),
        ("western_europe", western_europe),
        ("northern_europe", northern_europe),
    ]:
        os.makedirs(f"cache/era5/{region}", exist_ok=True)
        for month in [f"{m:02d}" for m in range(1, 13)]:
            target = Path(f"cache/era5/{region}/2019_{month}.nc")
            if not target.exists():
                if client is None:
                    client = cdsapi.Client()
                print(f"\n{target} does not exist; downloading...\n")
                client.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "format": "netcdf",
                        "variable": variables,
                        "year": "2019",
                        "month": month,
                        "day": [f"{d:02d}" for d in range(1, 32)],
                        "time": times,
                        "area": area,
                        "grid": grid,
                    },
                    target,
                )


if __name__ == "__main__":
    main()
