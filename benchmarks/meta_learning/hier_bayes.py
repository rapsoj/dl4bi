#!/usr/bin/env python
import importlib
from pathlib import Path
from typing import Callable

import hydra
import jax
import jax.numpy as jnp
import optax
import wandb
from jax import jit, random
from numpyro.infer import MCMC, NUTS
from omegaconf import DictConfig, OmegaConf

from dl4bi.meta_learning.train_utils import (
    cfg_to_run_name,
    cosine_annealing_lr,
    evaluate,
    instantiate,
    load_ckpt,
    save_ckpt,
    select_steps,
    train,
)

# TODO:
# Plot comparison images - pointwise post pred
# figure out good metrics to use
# Can you use distance bias on covariates too??
# Can we do House Electricity Consumption dataset?? (one million point GP paper)


@hydra.main("configs/hier_bayes", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    path = f"results/{cfg.project}/{cfg.seed}/{run_name}"
    path = Path(path)
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    if cfg.compare_inference:
        model_path = path.with_suffix(".ckpt")
        return compare_inference(rng, model_path, cfg)
    dataloader, *_ = collect_infer_funcs(cfg.inference_model, cfg.data)
    rng_train, rng_test = random.split(rng)
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
    train_step, valid_step = select_steps(model)
    state = train(
        rng_train,
        model,
        optimizer,
        train_step,
        valid_step,
        dataloader,
        dataloader,
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
    )
    metrics = evaluate(
        rng_test,
        state,
        valid_step,
        dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def collect_infer_funcs(model_name: str, data: DictConfig):
    module = importlib.import_module(f"inference_models.{model_name}")
    dataloader = module.build_dataloader(module.jax_prior_pred, data)
    return (
        dataloader,
        module.batch_to_infer_kwargs,
        module.numpyro_model,
        module.numpyro_pointwise_post_pred,
    )


def compare_inference(rng: jax.Array, model_path: Path, cfg: DictConfig):
    rng_sample, rng_extra, rng_mcmc, rng_post = random.split(rng, 4)
    dataloader, *infer_funcs = collect_infer_funcs(cfg.inference_model, cfg.data)
    state, _ = load_ckpt(model_path)
    batch = next(dataloader(rng_sample))
    s_ctx, f_ctx, _, s_test, f_test, *_ = batch
    Nc = cfg.infer.num_ctx
    f_mu_model, f_std_model = jit(state.apply_fn)(
        {"params": state.params, **state.kwargs},
        s_ctx[[0], :Nc],
        f_ctx[[0], :Nc],
        s_test[[0], Nc:],
        jnp.array([cfg.infer.num_ctx]),
        valid_lens_test=None,
        rngs={"extra": rng_extra},
    )
    batch_to_infer_kwargs, numpyro_model, numpyro_pointwise_post_pred = infer_funcs
    rng_sample, rng_mcmc, rng_post, rng = random.split(rng, 4)
    batch = next(dataloader(rng_sample))
    kwargs = batch_to_infer_kwargs(batch, cfg.data, cfg.infer)
    mcmc = run_mcmc(rng_mcmc, numpyro_model, cfg.infer.mcmc, **kwargs)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    f_mu_pyro, f_std_pyro = numpyro_pointwise_post_pred(
        rng_post,
        **kwargs,
        **samples,
    )
    print(f_mu_model.shape, f_mu_pyro.shape)


def run_mcmc(rng: jax.Array, model: Callable, infer: DictConfig, **kwargs):
    mcmc = MCMC(
        NUTS(model),
        num_warmup=infer.num_warmup,
        num_samples=infer.num_samples,
        num_chains=infer.num_chains,
    )
    mcmc.run(rng, **kwargs)
    return mcmc


if __name__ == "__main__":
    main()
