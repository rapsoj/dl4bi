import re
from math import prod
from pathlib import Path

import hydra
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from celeba import build_dataloaders as build_dataloaders_celeba
from cifar_10 import build_dataloaders as build_dataloaders_cifar_10
from jax import random
from mnist import build_dataloaders as build_dataloaders_mnist
from omegaconf import DictConfig

from dl4bi.meta_regression.train_utils import (
    build_gp_dataloader,
    load_ckpts,
    plot_img,
    plot_posterior_predictive,
)


@hydra.main(config_name="default", version_base=None)
def main(cfg: DictConfig):
    gp_tasks = re.compile("Gaussian Processes", re.IGNORECASE)
    img_tasks = re.compile("MNIST|CelebA|Cifar", re.IGNORECASE)
    only_regex = re.compile(cfg.get("only", ".*"), re.IGNORECASE)
    exclude_regex = re.compile(cfg.get("exclude", r"^$"), re.IGNORECASE)
    num_samples = cfg.get("num_samples", 16)
    results_dir = Path(f"results/{cfg.project}")
    if gp_tasks.match(cfg.project):
        results_dir /= f"{cfg.data.name}/{cfg.kernel.kwargs.kernel.func}/{cfg.seed}"
        ckpts = load_ckpts(results_dir, only_regex, exclude_regex)
        if "1d" in cfg.data.name:
            return plot_1d_gp_samples(cfg, ckpts, num_samples, results_dir)
        return plot_2d_gp_samples(cfg, ckpts, num_samples, results_dir)
    elif img_tasks.match(cfg.project):
        results_dir /= f"{cfg.seed}"
        ckpts = load_ckpts(results_dir, only_regex, exclude_regex)
        return plot_2d_img_samples(cfg, ckpts, num_samples, results_dir)


def plot_1d_gp_samples(
    cfg: DictConfig,
    ckpts: dict,
    num_samples: int = 16,
    results_dir: Path = Path("."),
    num_ctx: int = 10,
    lengthscales: list[float] = [0.1, 0.3, 0.5],
):
    rng = random.key(cfg.seed)
    rng_data, rng_dropout, rng_extra = random.split(rng, 3)
    cfg.data.batch_size = 1
    cfg.data.num_ctx.min = num_ctx
    cfg.data.num_ctx.max = num_ctx
    cfg.kernel.kwargs.ls.kwargs.dist = "fixed"
    num_rows, num_cols = len(ckpts), len(lengthscales)
    fig_size = (6 * num_cols, 4 * num_rows)
    for i in range(num_samples):
        fig, axs = plt.subplots(num_rows, num_cols, figsize=fig_size)
        rng_i, rng_data = random.split(rng_data)
        for col, ls in enumerate(lengthscales):
            cfg.kernel.kwargs.ls.kwargs.kwargs = {"value": ls}
            dataloader = build_gp_dataloader(cfg.data, cfg.kernel)
            s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, *_ = next(dataloader(rng_i))
            for row, run_name in enumerate(sorted(ckpts)):
                state = ckpts[run_name]["state"]
                f_mu, f_std, *_ = state.apply_fn(
                    {"params": state.params, **state.kwargs},
                    s_ctx,
                    f_ctx,
                    s_test,
                    valid_lens_ctx,
                    rngs={"dropout": rng_dropout, "extra": rng_extra},
                )
                plt.sca(axs[row, col])
                plot_posterior_predictive(
                    s_ctx[0, :num_ctx, 0],
                    f_ctx[0, :num_ctx, 0],
                    s_test[0, :, 0],
                    f_test[0, :, 0],
                    f_mu[..., 0],
                    f_std[..., 0],
                )
                # NOTE: only set labels for non-interior plots
                if row == 0:
                    axs[row, col].set_title(f"ls={ls:0.1f}")
                if col == 0:
                    axs[row, col].set_ylabel(run_name)
                if row == num_rows - 1:
                    axs[row, col].set_xlabel("s")
        fig.tight_layout()
        fig.savefig(results_dir / f"comparison_sample_{i+1}.png", dpi=150)
        plt.clf()


def plot_2d_gp_samples(
    cfg: DictConfig,
    ckpts: dict,
    num_samples: int = 16,
    results_dir: Path = Path("."),
    num_ctx: int = 128,
):
    rng = random.key(cfg.seed)
    cfg.data.batch_size = 1
    cfg.data.num_ctx.min = 0  # use only grid points in dataloader
    cfg.data.num_ctx.max = 0
    cfg.kernel.kwargs.ls.kwargs.dist = "fixed"
    num_rows, num_cols = len(ckpts), 4
    fig_size = (6 * num_cols, 4 * num_rows)
    L = prod([axis.num for axis in cfg.data.s])
    for i in range(num_samples):
        fig, axs = plt.subplots(num_rows, num_cols, figsize=fig_size)
        rng_data, rng_permute, rng_dropout, rng_extra, rng = random.split(rng, 5)
        for col in range(num_cols):
            dataloader = build_gp_dataloader(cfg.data, cfg.kernel)
            _, _, _, s, f, *_ = next(dataloader(rng_data))
            permute_idx = random.choice(rng_permute, L, (L,), replace=False)
            inv_permute_idx = jnp.argsort(permute_idx)
            # permute the order and select the first valid_lens_ctx for context
            s_permuted = s[:, permute_idx, :]
            f_permuted = f[:, permute_idx, :]
            valid_lens_ctx = jnp.array([num_ctx])
            for row, run_name in enumerate(sorted(ckpts)):
                state = ckpts[run_name]["state"]
                f_mu, f_std, *_ = state.apply_fn(
                    {"params": state.params, **state.kwargs},
                    s_permuted,
                    f_permuted,
                    s_permuted,
                    valid_lens_ctx,
                    rngs={"dropout": rng_dropout, "extra": rng_extra},
                )
                plt.sca(axs[row, col])
                plt.clf()
                # TODO(danj): PLOT
        fig.tight_layout()
        fig.savefig(results_dir / f"comparison_sample_{i+1}.png", dpi=150)
        plt.clf()


def plot_2d_img_samples(
    cfg: DictConfig,
    ckpts: dict,
    num_samples: int = 16,
    results_dir: Path = Path("."),
    num_ctx: int = 50,
):
    matches = lambda pattern: re.match(pattern, cfg.project, re.IGNORECASE)
    # example project names include "TNP-KR - MNIST", "MNIST", etc, so match
    cmap = mpl.colormaps.get_cmap("Spectral_r")
    cmap.set_bad("grey")
    match cfg.project:
        case _ if matches("MNIST"):
            H, W, C = (28, 28, 1)
            build_dataloaders = build_dataloaders_mnist
            cmap = mpl.colormaps.get_cmap("grey")
            cmap.set_bad("blue")
        case _ if matches("CelebA"):
            H, W, C = (32, 32, 3)
            build_dataloaders = build_dataloaders_celeba
        case _ if matches("Cifar"):
            H, W, C = (32, 32, 3)
            build_dataloaders = build_dataloaders_cifar_10
        case _:
            raise Exception(f"No dataloader defined for {cfg.project}!")
    train_dataloader, *_ = build_dataloaders(
        batch_size=1,
        num_ctx_min=num_ctx,
        num_ctx_max=num_ctx,
        num_test_max=H * W,
    )
    rng = random.key(cfg.seed)
    num_models = len(ckpts)
    for i in range(num_samples):
        rng_dropout, rng_extra, rng_i, rng = random.split(rng, 4)
        (
            s_ctx,
            f_ctx,
            valid_lens_ctx,
            _,
            _,
            valid_lens_test,
            s_test_full,
            f_test_full,
            inv_permute_idx,
        ) = next(train_dataloader(rng_i))
        num_rows, num_cols = num_models, 4
        fig, axs = plt.subplots(
            num_rows,
            num_cols,
            figsize=(6 * num_cols, 6 * num_rows),
        )
        for row_idx, model_name in enumerate(sorted(ckpts)):
            state = ckpts[model_name]["state"]
            f_mu, f_std, *_ = state.apply_fn(
                {"params": state.params, **state.kwargs},
                s_ctx,
                f_ctx,
                s_test_full,
                valid_lens_ctx,
                valid_lens_test,
                rngs={"dropout": rng_dropout, "extra": rng_extra},
            )
            plot_img(
                i,
                (H, W, C),
                f_ctx[0, :num_ctx],
                f_mu[0],
                f_std[0],
                f_test_full[0],
                inv_permute_idx,
                axs[row_idx],
                cmap=cmap,
            )
            # NOTE: unset titles by row, unless its the first row
            for col_idx in range(num_cols):
                axs[row_idx, col_idx].set_xticks([])
                axs[row_idx, col_idx].set_yticks([])
                if row_idx != 0:
                    axs[row_idx, col_idx].set_title("")
                else:
                    title = axs[row_idx, col_idx].get_title()
                    axs[row_idx, col_idx].set_title(title, fontsize=36)
            axs[row_idx, 0].set_ylabel(f"{model_name}", fontsize=36)
        fig.tight_layout()
        fig.savefig(results_dir / f"comparison_sample_{i+1}.png", dpi=150)
        plt.clf()


if __name__ == "__main__":
    main()
