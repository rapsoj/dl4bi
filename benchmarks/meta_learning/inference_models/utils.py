import importlib
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, random
from jax.scipy.stats import norm
from numpyro.infer import MCMC, NUTS
from omegaconf import DictConfig
from sps.kernels import rbf


def collect_infer_funcs(infer_model_name: str, data: DictConfig):
    module = importlib.import_module(f"inference_models.{infer_model_name}")
    prior_pred = getattr(module, "jax_prior_pred", module.numpyro_prior_pred)
    dataloader = module.build_dataloader(prior_pred, data)
    return (
        dataloader,
        module.batch_to_infer_kwargs,
        module.numpyro_model,
        module.numpyro_pointwise_post_pred,
        module.visualize,
    )


# TODO(danj): verify this formula
@partial(jit, static_argnames=("jitter", "kernel"))
def condition_gp(
    rng: jax.Array,
    s_ctx: jax.Array,
    f_ctx: jax.Array,
    s_test: jax.Array,
    var: jax.Array,
    ls: jax.Array,
    kernel: Callable = rbf,
    jitter=1e-5,
):
    """Conditions a GP on observed points.

    Source: https://num.pyro.ai/en/stable/examples/gp.html
    """
    L_ctx, L_test = s_ctx.shape[0], s_test.shape[0]
    k_tt = kernel(s_test, s_test, var, ls) + jitter * jnp.eye(L_test)
    k_tc = kernel(s_test, s_ctx, var, ls)
    k_cc = kernel(s_ctx, s_ctx, var, ls) + jitter * jnp.eye(L_ctx)
    K_cc_cho = jax.scipy.linalg.cho_factor(k_cc)
    K = k_tt - k_tc @ jax.scipy.linalg.cho_solve(K_cc_cho, k_tc.T)
    mu = k_tc @ jax.scipy.linalg.cho_solve(K_cc_cho, f_ctx)
    noise = jnp.sqrt(jnp.clip(jnp.diag(K), 0.0)) * random.normal(rng, L_test)
    return mu + noise


def run_mcmc(rng: jax.Array, model: Callable, infer: DictConfig, **kwargs):
    mcmc = MCMC(
        NUTS(model),
        num_warmup=infer.num_warmup,
        num_samples=infer.num_samples,
        num_chains=infer.num_chains,
    )
    mcmc.run(rng, **kwargs)
    return mcmc


def visualize_spatial(**kwargs):
    if kwargs["s_ctx"].ndim == 1:
        plot_1d_posterior_predictive(**kwargs)
    else:  # 2D
        plot_2d_posterior_predictive(**kwargs)


# TODO(danj): update to use a real path
def plot_1d_posterior_predictive(
    s_ctx: jax.Array,  # [L_ctx]
    f_ctx: jax.Array,  # [L_ctx]
    s: jax.Array,  # [L]
    f: jax.Array,  # [L]
    f_mu_model: jax.Array,  # [L]
    f_std_model: jax.Array,  # [L]
    f_mu_pyro: jax.Array,  # [L]
    f_std_pyro: jax.Array,  # [L]
    hdi_prob: float = 0.95,
):
    # palette from https://davidmathlogic.com/colorblind
    magenta, green, blue, gold = "#D81B60", "#D81B60", "#1E88E5", "#FFC107"
    plt.scatter(s_ctx, f_ctx, color=magenta)
    plt.plot(s, f, color=magenta)
    _plot_1d_bounds(s, f_mu_model, f_std_model, color=blue)
    _plot_1d_bounds(s, f_mu_pyro, f_std_pyro, color=gold)
    plt.xlabel("s")
    plt.ylabel("f")
    plt.title("GP 1D")
    plt.savefig("/tmp/test.pdf")
    return "/tmp/test.pdf"


def _plot_1d_bounds(
    s: jax.Array,
    f_mu: jax.Array,
    f_std: jax.Array,
    color: str = "steelblue",
    hdi_prob: float = 0.95,
):
    z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
    f_lower = f_mu - z_score * f_std
    f_upper = f_mu + z_score * f_std
    plt.plot(s, f_mu, color=color)
    plt.fill_between(s, f_lower, f_upper, alpha=0.4, color=color, interpolate=True)


def plot_2d_posterior_predictive(
    s_ctx: jax.Array,  # [L_ctx, 2]
    f_ctx: jax.Array,  # [L_ctx]
    s: jax.Array,  # [L, 2]
    f: jax.Array,  # [L]
    f_mu_model: jax.Array,  # [L]
    f_std_model: jax.Array,  # [L]
    f_mu_pyro: jax.Array,  # [L]
    f_std_pyro: jax.Array,  # [L]
    hdi_prob: float = 0.95,
):
    # TODO(danj): implement
    pass


# TODO(danj): complete
def compute_metrics(**kwargs):
    return {}
