import sys

sys.path.append("benchmarks/vae")

from pathlib import Path
from typing import Callable, Optional, Union

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import imageio

import numpyro
import optax
from jax import Array, jit, random
from jax.nn import sigmoid
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from sps.utils import build_grid # NOTE: this is some stochastic simulation package not in the repo
from utils.plot_utils import plot_infer_trace

import wandb
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import cosine_annealing_lr, train
from dl4bi.vae import gMLPDeepRV
from dl4bi.vae.train_utils import deep_rv_train_step, generate_surrogate_decoder
from deep_rv_kernels import (
    separable_kernel_family,
    nonsep_kernel_family,
    advected_kernel_family,
)

from numpyro.distributions import Distribution
from numpyro.distributions import constraints

import pickle
import json

class GeneralizedPareto(Distribution):
    arg_constraints = {
        "scale": constraints.positive,
        "concentration": constraints.real,
    }
    support = constraints.nonnegative

    def __init__(self, scale, concentration, validate_args=None):
        self.scale = scale
        self.concentration = concentration
        super().__init__(
            batch_shape=jnp.shape(scale),
            event_shape=(),
            validate_args=validate_args,
        )

    def log_prob(self, value):
        sigma = self.scale
        xi = self.concentration

        z = 1 + xi * value / sigma
        safe_z = jnp.maximum(z, 1e-12)

        log_pdf = jnp.where(
            jnp.abs(xi) > 1e-6,
            -jnp.log(sigma) - (1.0 / xi + 1.0) * jnp.log(safe_z),
            -jnp.log(sigma) - value / sigma,
        )

        valid = (value >= 0) & (z > 0)
        return jnp.where(valid, log_pdf, -jnp.inf)

    def sample(self, key, sample_shape=()):
        sigma = self.scale
        xi = self.concentration
        u = random.uniform(key, sample_shape + sigma.shape)

        # inverse CDF sampling
        return jnp.where(
            jnp.abs(xi) > 1e-6,
            sigma / xi * (jnp.power(u, -xi) - 1.0),
            -sigma * jnp.log(u),
        )


def compute_threshold(y_train: Array, k: float = 3.0) -> float: ### CHANGE: Helper to compute threshold from sample
    """
    Compute threshold u = mean(y_train) + k * std(y_train).
    Accepts y_train of any shape; flattens it.
    """
    flat = jnp.ravel(y_train)
    mean = jnp.mean(flat)
    std = jnp.std(flat)
    return (mean + k * std).item()

def save_comparison_gif(y_true, y_pred, grid_size, T, path):
    frames = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for t in range(T):
        true_frame = y_true[t::T].reshape(grid_size, grid_size)
        pred_frame = y_pred.mean(axis=0)[t::T].reshape(grid_size, grid_size)

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        vmin = min(y_true.min(), y_pred.mean(axis=0).min())
        vmax = max(y_true.max(), y_pred.mean(axis=0).max())

        axs[0].imshow(true_frame, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axs[0].set_title("True")
        axs[0].axis("off")

        axs[1].imshow(pred_frame, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axs[1].set_title("Pred")
        axs[1].axis("off")

        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        frames.append(image)
        plt.close(fig)

    imageio.mimsave(path, frames, fps=3)


### CHANGE: To accept custom k for defining threshold
def main(seed=57, gt_ls=20, k: float = 3.0, kernel_family: str ='sep'):
    # kernel_family = {sep, nonsep, advected}
    if kernel_family == "sep":
        kernel_family = separable_kernel_family
    elif kernel_family == "nonsep":
        kernel_family = nonsep_kernel_family
    elif kernel_family == "advected":
        kernel_family = advected_kernel_family
    else:
        raise ValueError("Unknown kernel_family")
    # NOTE: generate seeds and directories.
    rng = random.key(seed)
    rng_train, rng_infer, rng_idxs, rng_obs, rng = random.split(rng, 5)
    wandb.init(mode="disabled")
    save_dir = Path("results/DeepRV_example/")
    save_dir.mkdir(parents=True, exist_ok=True)
    # NOTE: generates the spatial grid to train and infer on
    s = build_grid([{"start": 0.0, "stop": 100.0, "num": 12}] * 2).reshape(-1, 2)
    # CHANGE: generate spatio-temporal grid based on spatial grid
    T = 6
    t_vals = jnp.linspace(0, 1, T)

    s_rep = jnp.repeat(s, T, axis=0)
    t_rep = jnp.tile(t_vals, s.shape[0]).reshape(-1, 1)

    s_st = jnp.concatenate([s_rep, t_rep], axis=1)

    # Create synthetic training data first (so we can derive threshold from training set)
    # For demonstration we treat gen_y_obs output as the training data.
    # Compute threshold from training data y (if you have a real training set, pass it here)
    # Generate an initial dataset to compute u
    rng_train_data, rng_obs_data = random.split(rng_obs)
    y_train, mu_train, sigma_train = gen_y_obs(rng_train_data, s_st, gt_ls, u=10.0, p_exceed=0.15, kernel_family=kernel_family)
    u = compute_threshold(y_train, k=k)

    # NOTE: The observed outcome to perform inference on
    # Generate final observed data using the computed threshold
    y_obs, mu_true, sigma_true = gen_y_obs(rng_obs_data, s_st, gt_ls, u=u, p_exceed=0.15, kernel_family=kernel_family)
    xi_true = 0.1
    p_true = 0.15

    # NOTE: Priors for training and inference
    priors = {"ls": dist.Uniform(1.0, 100.0), "beta": dist.Normal()}
    sqrt_N = int(jnp.sqrt(s.shape[0]))
    # NOTE: Mask detailing which locations are observable
    obs_mask_spatial = gen_spatial_obs_mask(rng_idxs, (sqrt_N, sqrt_N), obs_ratio=0.7)

    # CHANGE: temporal dropout (e.g. 80% observed per timestep)
    rng_idxs, rng_t = random.split(rng_idxs)
    temporal_mask = random.bernoulli(rng_t, 0.8, (obs_mask_spatial.shape[0], T))

    obs_mask = (obs_mask_spatial[:, None] & temporal_mask).reshape(-1)
    infer_model = inference_model(s_st, priors, u=u, kernel_family=kernel_family) # CHANGE: make input spatio-temporal
    # NOTE: surrogate training
    nn_model = gMLPDeepRV(num_blks=2)
    optimizer = optax.adamw(cosine_annealing_lr(100_000, 1e-3), weight_decay=1e-2)
    optimizer = optax.chain(optax.clip_by_global_norm(3.0), optimizer)
    loader = gen_train_dataloader(s_st, priors, kernel_family) # CHANGE: make input spatio-temporal
    state = train(
        rng_train,
        nn_model,
        optimizer,
        deep_rv_train_step,
        100_000,
        loader,
        valid_step,
        25_000,
        5_000,
        loader,
        return_state="best",
        valid_monitor_metric="norm MSE",
    )
    surrogate_decoder = generate_surrogate_decoder(state, nn_model)
    # NOTE: Inference DeepRV
    samples_drv, mcmc_drv, y_hat_drv = hmc(
        rng_infer, infer_model, y_obs, obs_mask, surrogate_decoder
    )
    cond_names = list(priors.keys())
    # NOTE: Plotting inference traces, and mean predictions
    plot_infer_trace(
        samples_drv, mcmc_drv, None, cond_names, save_dir / "infer_trace_drv.png"
    )
    plot_models_predictive_means(
        sqrt_N, y_obs, [y_hat_drv], obs_mask, ["DeepRV"], save_dir / "obs_means.png", u
    )

    save_comparison_gif(
        y_obs,
        y_hat_drv["mu"],  # or use excess if preferred
        sqrt_N,
        T,
        save_dir / "comparison.gif"
    )
    #### CHANGE: Evaluate performance on extreme data
    evaluate_extremes(
        samples_drv,
        y_hat_drv,
        y_obs,
        u,
        xi_true,
        p_true,
        sigma_true,
        mcmc_drv,
        save_dir,
    )

    #### CHANGE: Save model
    with open(save_dir / "surrogate.pkl", "wb") as f:
        pickle.dump(state.params, f)


def hmc(
    rng: Array,
    model: Callable,
    y_obs: Array,
    obs_mask: Union[bool, Array],
    surrogate_decoder: Optional[Callable] = None,
):
    """runs HMC on given inference model and observed f"""
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = random.split(rng)
    mcmc = MCMC(nuts, num_chains=2, num_samples=1_000, num_warmup=1_000)
    mcmc.run(k1, surrogate_decoder=surrogate_decoder, obs_mask=obs_mask, y=y_obs)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    post = Predictive(model, samples)(k2, surrogate_decoder=surrogate_decoder)

    return samples, mcmc, post

# CHANGE: add kernel function argument, now can take different kernel functions
def gen_train_dataloader(s: Array, priors: dict, kernel_family, batch_size=32):
    jitter = 1e-3 * jnp.eye(s.shape[0])

    f_jit = jit(lambda L, z: jnp.einsum("ij,bj->bi", L, z))

    def dataloader(rng_data):
        while True:
            rng_data, rng_params, rng_z = random.split(rng_data, 3)

            # sample kernel parameters
            params = kernel_family.sample_params(rng_params)

            # build covariance
            K = kernel_family.compute(s, s, params)
            K = 0.5 * (K + K.T) + jitter

            L = jnp.linalg.cholesky(K)

            # sample latent with random walk for temporal dynamics
            T = jnp.unique(s[:, -1]).shape[0]
            N = s.shape[0] // T

            z_eps = dist.Normal().sample(rng_z, (batch_size, T, N))
            z_t = jnp.cumsum(z_eps, axis=1)
            z = z_t.reshape(batch_size, -1)

            yield {
                "s": s,
                "z": z,
                "conditionals": kernel_family.build_conditionals(params),
                "f": f_jit(L, z),
            }

    return dataloader

### CHANGE: Replace with POT model (Bernoulli + GPD for excesses)
def inference_model(s: Array, priors: dict, u: float, kernel_family):
    """
    Builds POT inference model:
     - Bernoulli for exceedance indicator (global probability)
     - GPD for excesses, with spatial scale sigma = exp(beta + mu)
     - constant xi (constrained)
    """
    surrogate_kwargs = {"s": s}

    def gpd_pot(surrogate_decoder=None, obs_mask=True, y=None):
        # hyperpriors
        var = 1.0
        ls = numpyro.sample("ls", priors["ls"])
        beta = numpyro.sample("beta", priors["beta"])

        # Occurrence (global probability) prior on logit scale
        pi_logit = numpyro.sample("pi_logit", dist.Normal(0.0, 1.0))
        p = jnp.clip(sigmoid(pi_logit), 1e-6, 1 - 1e-6)

        # CHANGE: moving from spatio-temporal smoothing to forecasting with latent state evolution
        # latent z with temporal dynamics (random walk)
        T = jnp.unique(s[:, -1]).shape[0] # inferred from input
        N = s.shape[0] // T
        z_eps = numpyro.sample("z_eps", dist.Normal(), sample_shape=(T, N))
        # random walk in time
        z_t = jnp.cumsum(z_eps, axis=0)  # shape (T, N)
        # flatten to match DeepRV input
        z = z_t.reshape(1, -1)
        params = kernel_family.sample_params(random.key(0))
        params["ls_space"] = ls

        if surrogate_decoder is None:
            # CHANGE: spatial structure from GP, temporal structure from random walk
            # aka apply GP per time slice
            T = jnp.unique(s[:, -1]).shape[0]
            N = s.shape[0] // T

            mu_list = []

            for t in range(T):
                idx = jnp.arange(t, s.shape[0], T)

                K_t = kernel_family.compute(s[idx], s[idx], params)
                K_t = 0.5 * (K_t + K_t.T) + 1e-3 * jnp.eye(N)

                L_t = jnp.linalg.cholesky(K_t)
                mu_t = jnp.matmul(L_t, z_t[t])

                mu_list.append(mu_t)

            mu = numpyro.deterministic("mu", jnp.concatenate(mu_list))
        else:
            mu = numpyro.deterministic(
                "mu",
                surrogate_decoder(
                    z,
                    kernel_family.build_conditionals(params),
                    **surrogate_kwargs
                ).squeeze()
            )

        sigma = jnp.exp(beta + mu)  # scale (>0)

        # stable constrained xi: map raw to (-0.5, 0.5) for stability
        xi_raw = numpyro.sample("xi_raw", dist.Normal(0.0, 0.5))
        xi = 0.5 * jnp.tanh(xi_raw)
        numpyro.deterministic("xi", xi)

        # derive exceedance indicator and excess values
        if y is not None:
            exceed = (y > u)
            excess = y - u # non-exceedances -> zero but will be masked
        else:
            exceed = None
            excess = None

        # 1) Model occurrence (Bernoulli) at observed locations
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample(
                "occ",
                dist.Bernoulli(probs=p).expand([s.shape[0]]),
                obs=None if y is None else exceed.astype(jnp.int32),
            )

        # 2) Model excesses only where exceedances occur and are observed
        if y is not None:
            mask_excess = (exceed & obs_mask)
        else:
            mask_excess = obs_mask  # or True

        with numpyro.handlers.mask(mask=mask_excess):
            numpyro.sample(
                "excess",
                GeneralizedPareto(scale=sigma, concentration=xi),
                obs=None if y is None else excess,
            )

    return gpd_pot


@jit
def valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    metrics = output.metrics(batch["f"], 1.0)
    return {"norm MSE": metrics["MSE"]}

### CHANGE: New function to evaluation performance on extreme values
def evaluate_extremes(samples, predictive, y_obs, u,
                      xi_true, p_true, sigma_true,
                      mcmc_obj, save_dir):

    results = {}

    xi_samples = np.array(samples["xi"])
    xi_mean = xi_samples.mean()
    xi_ci = np.percentile(xi_samples, [2.5, 97.5])

    results["xi_bias"] = float(xi_mean - xi_true)
    results["xi_covered"] = bool(xi_ci[0] <= xi_true <= xi_ci[1])

    pi_logit = np.array(samples["pi_logit"])
    p_samples = 1 / (1 + np.exp(-pi_logit))
    p_mean = p_samples.mean()
    p_ci = np.percentile(p_samples, [2.5, 97.5])

    results["p_bias"] = float(p_mean - p_true)
    results["p_covered"] = bool(p_ci[0] <= p_true <= p_ci[1])

    # ----- Return levels -----
    T = 100

    beta_samples = np.array(samples["beta"])
    mu_samples = np.array(samples["mu"])
    xi_samples = np.array(samples["xi"])

    sigma_samples = np.exp(beta_samples[:, None] + mu_samples)

    zT_samples = u + (sigma_samples / xi_samples[:, None]) * (
        (T * p_samples[:, None]) ** xi_samples[:, None] - 1
    )

    zT_mean = zT_samples.mean(axis=0)

    zT_true = u + (sigma_true / xi_true) * (
        (T * p_true) ** xi_true - 1
    )

    rmse = np.sqrt(((zT_mean - zT_true) ** 2).mean())

    zT_lower = np.percentile(zT_samples, 2.5, axis=0)
    zT_upper = np.percentile(zT_samples, 97.5, axis=0)

    coverage = np.mean((zT_true >= zT_lower) &
                       (zT_true <= zT_upper))
    results["return_level_rmse"] = float(rmse)
    results["return_level_coverage"] = float(coverage)

    results["num_divergences"] = int(
        mcmc_obj.get_extra_fields()["diverging"].sum()
    )

    with open(save_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    np.savez(
        save_dir / "posterior_samples.npz",
        **{k: np.array(v) for k, v in samples.items()}
    )


### CHANGE: Replace with a generator that creates exceedances and non-exceedances
def gen_y_obs(rng: Array, s: Array, gt_ls: float, u: float, p_exceed: float = 0.15, kernel_family=None):
    """
    Generates synthetic observations with some exceedances over threshold u.
    - p_exceed: true probability of exceedance (global) used to generate dataset
    Returns y_obs (shape N,)
    """
    rng_mu, rng_occ, rng_gpd, rng_base = random.split(rng, 4)
    var, ls, beta = 1.0, gt_ls, 1.0

    rng_params, rng_mu = random.split(rng_mu)
    params = kernel_family.sample_params(rng_params)
    params["ls_space"] = gt_ls  # keep ground truth control

    # CHANGE: replace matern 1/2 with st kernel
    K = kernel_family.compute(s, s, params)
    K = 0.5 * (K + K.T) + 1e-3 * jnp.eye(s.shape[0])

    N = s.shape[0]
    mu = dist.MultivariateNormal(jnp.zeros(N), K).sample(rng_mu)

    sigma = jnp.exp(beta + mu)  # scale for GPD at each location
    xi_true = 0.1  # ground truth shape

    # occurrence indicators
    occ = dist.Bernoulli(probs=p_exceed).sample(rng_occ, (s.shape[0],))

    # sample excesses where occ == 1, else sample baseline values < u
    # For baseline non-exceedances, use a simple Uniform(0, u) so they are < u
    # Create arrays of draws:
    excess_samples = GeneralizedPareto(
        scale=sigma,
        concentration=xi_true,
    ).sample(rng_gpd)
    base_samples = dist.Uniform(0.0, u * 0.9).sample(rng_base, sample_shape=(s.shape[0],))

    y = jnp.where(occ.astype(jnp.bool_), u + excess_samples, base_samples)
    return y, mu, sigma


def gen_spatial_obs_mask(rng: Array, grid_shape: tuple, obs_ratio: float = 0.15):
    """
    Generates a spatial observation mask for a 2D grid. Keeps a certain percentage of the domain unmasked,
    in the form of a few spatially-contiguous elliptical blobs. The output is a 1D boolean mask indicating
    which locations are observed.

    Args:
        rng: JAX PRNG key
        y_obs: Flattened signal (N,)
        grid_shape: Tuple (H, W) for reshaping the 1D signal
        obs_ratio: Fraction of the total grid to remain observed

    Returns:
        mask_flat: Flattened boolean mask of shape (N,), where True = observed, False = masked
    """
    H, W = grid_shape
    total_points = H * W
    num_obs_points = int(obs_ratio * total_points)
    mask = jnp.zeros((H, W), dtype=bool)

    points_collected = 0
    blob_idx = 0
    while points_collected < num_obs_points:
        rng_blob, rng = random.split(rng)
        rngs = random.split(rng_blob, 4)
        center_x = random.randint(rngs[0], (), 0, H)
        center_y = random.randint(rngs[1], (), 0, W)
        radius_x = random.randint(rngs[2], (), H // 8, H // 4)
        radius_y = random.randint(rngs[3], (), W // 8, W // 4)
        yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
        ellipse = (
            ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2
        ) <= 1.0
        new_mask = jnp.logical_or(mask, ellipse)
        added = jnp.sum(new_mask) - jnp.sum(mask)
        mask = new_mask
        points_collected += int(added)
        blob_idx += 1
    # NOTE: If we overshot, randomly drop extras
    if points_collected > num_obs_points:
        flat_idxs = jnp.argwhere(mask.flatten()).squeeze()
        rng_trim, _ = random.split(rngs[-1])
        selected = random.choice(
            rng_trim, flat_idxs, shape=(num_obs_points,), replace=False
        )
        final_mask = jnp.zeros(total_points, dtype=bool).at[selected].set(True)
    else:
        final_mask = mask.flatten()

    return final_mask

### CHANGE: Plot POTS (observed exceedance map, posterior mean/median probability of exceedance)
def plot_models_predictive_means(
        grid_size, y_obs, predictive_list, obs_mask, model_names, save_path: Path, u: float):
    """
    predictive_list: list of posterior predictive dicts returned by Predictive(model, samples)
    Each predictive dict must contain keys "occ" and "excess".
    """
    pred = predictive_list[0]  # only DeepRV case here
    # posterior predicted occurrence probability: mean over posterior draws
    occ_samples = pred["occ"]  # shape (num_samples, N)

    # posterior predicted excess median (set zeros where no exceed)
    excess_samples = pred["excess"]  # shape (num_samples, N) with zeros/masked where not exceed
    N = grid_size * grid_size
    occ_mean = jnp.mean(occ_samples, axis=0)[:N].reshape(grid_size, grid_size)
    excess_median = jnp.median(excess_samples, axis=0)[:N].reshape(grid_size, grid_size)

    y_obs_t0 = y_obs[:N].reshape(grid_size, grid_size)
    observed_exceed = (y_obs_t0 > u).astype(float)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(observed_exceed, origin="lower", cmap="viridis")
    axs[0].set_title("Observed exceedances")

    im1 = axs[1].imshow(occ_mean, origin="lower", cmap="viridis", vmin=0, vmax=1)
    axs[1].set_title("Posterior exceedance prob (mean)")
    fig.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(excess_median, origin="lower", cmap="viridis")
    axs[2].set_title("Posterior excess median")
    fig.colorbar(im2, ax=axs[2])

    fig.savefig(save_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()


# TODO:
# Run for different simulations
# Run on real-world data

# EXAMPLE:
# main(kernel_family="sep")
# main(kernel_family="nonsep")
# main(kernel_family="advected")