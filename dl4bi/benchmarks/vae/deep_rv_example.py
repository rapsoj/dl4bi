import sys

sys.path.append("benchmarks/vae")

from pathlib import Path
from typing import Callable, Optional, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt

import numpyro
import optax
from jax import Array, jit, random
from jax.nn import sigmoid
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from sps.kernels import matern_1_2 # NOTE: this is some stochastic simulation package not in the repo
from sps.utils import build_grid # NOTE: this is some stochastic simulation package not in the repo
from utils.plot_utils import plot_infer_trace

import wandb
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import cosine_annealing_lr, train
from dl4bi.vae import gMLPDeepRV
from dl4bi.vae.train_utils import deep_rv_train_step, generate_surrogate_decoder

from numpyro.distributions import Distribution
from numpyro.distributions import constraints
from jax import lax


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

### CHANGE: To accept custom k for defining threshold
def main(seed=57, gt_ls=20, k: float = 3.0):
    # NOTE: generate seeds and directories.
    rng = random.key(seed)
    rng_train, rng_infer, rng_idxs, rng_obs, rng = random.split(rng, 5)
    wandb.init(mode="disabled")
    save_dir = Path("results/DeepRV_example/")
    save_dir.mkdir(parents=True, exist_ok=True)
    # NOTE: generates the spatial grid to train and infer on
    s = build_grid([{"start": 0.0, "stop": 100.0, "num": 16}] * 2).reshape(-1, 2)

    # Create synthetic training data first (so we can derive threshold from training set)
    # For demonstration we treat gen_y_obs output as the training data.
    # Compute threshold from training data y (if you have a real training set, pass it here)
    # Generate an initial dataset to compute u
    rng_train_data, rng_obs_data = random.split(rng_obs)
    y_train = gen_y_obs(rng_train_data, s, gt_ls, u=10.0, p_exceed=0.15)
    u = compute_threshold(y_train, k=k)

    # NOTE: The observed outcome to perform inference on
    # Generate final observed data using the computed threshold
    y_obs = gen_y_obs(rng_obs_data, s, gt_ls, u=u, p_exceed=0.15)

    # NOTE: Priors for training and inference
    priors = {"ls": dist.Uniform(1.0, 100.0), "beta": dist.Normal()}
    sqrt_N = int(jnp.sqrt(s.shape[0]))
    # NOTE: Mask detailing which locations are observable
    obs_mask = gen_spatial_obs_mask(rng_idxs, (sqrt_N, sqrt_N), obs_ratio=0.7)
    infer_model = inference_model(s, priors, u=u)
    # NOTE: surrogate training
    nn_model = gMLPDeepRV(num_blks=2)
    optimizer = optax.adamw(cosine_annealing_lr(100_000, 1e-3), weight_decay=1e-2)
    optimizer = optax.chain(optax.clip_by_global_norm(3.0), optimizer)
    loader = gen_train_dataloader(s, priors)
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


def gen_train_dataloader(s: Array, priors: dict, batch_size=32):
    jitter = 5e-4 * jnp.eye(s.shape[0])
    kernel_jit = jit(lambda s, var, ls: matern_1_2(s, s, var, ls) + jitter)
    f_jit = jit(lambda L, z: jnp.einsum("ij,bj->bi", L, z))

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_z = random.split(rng_data, 3)
            var = 1.0
            ls = priors["ls"].sample(rng_ls)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, s.shape[0]))
            K = kernel_jit(s, var, ls)
            L = jnp.linalg.cholesky(K)
            yield {"s": s, "z": z, "conditionals": jnp.array([ls]), "f": f_jit(L, z)}

    return dataloader

### CHANGE: Replace with POT model (Bernoulli + GPD for excesses)
def inference_model(s: Array, priors: dict, u: float):
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

        # latent z for DeepRV / GP reparam
        z = numpyro.sample("z", dist.Normal(), sample_shape=(1, s.shape[0]))

        if surrogate_decoder is None:
            K = matern_1_2(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
            L_chol = jnp.linalg.cholesky(K)
            mu = numpyro.deterministic("mu", jnp.matmul(L_chol, z[0]))
        else:
            mu = numpyro.deterministic(
                "mu", surrogate_decoder(z, jnp.array([ls]), **surrogate_kwargs).squeeze()
            )

        sigma = jnp.exp(beta + mu)  # scale (>0)

        # stable constrained xi: map raw to (-0.5, 0.5) for stability
        xi_raw = numpyro.sample("xi_raw", dist.Normal(0.0, 0.5))
        xi = 0.5 * jnp.tanh(xi_raw)
        numpyro.deterministic("xi", xi)

        # derive exceedance indicator and excess values
        exceed = (y > u)
        excess = y - u  # non-exceedances -> zero but will be masked

        # 1) Model occurrence (Bernoulli) at observed locations
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample(
                "occ",
                dist.Bernoulli(probs=p).expand([s.shape[0]]),
                obs=exceed.astype(jnp.int32),
            )

        # 2) Model excesses only where exceedances occur and are observed
        mask_excess = (exceed & obs_mask).astype(bool)

        with numpyro.handlers.mask(mask=mask_excess):
            numpyro.sample(
                "excess",
                GeneralizedPareto(scale=sigma, concentration=xi),
                obs=excess,
            )

    return gpd_pot


@jit
def valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    metrics = output.metrics(batch["f"], 1.0)
    return {"norm MSE": metrics["MSE"]}

### CHANGE: Replace with a generator that creates exceedances and non-exceedances
def gen_y_obs(rng: Array, s: Array, gt_ls: float, u: float, p_exceed: float = 0.15):
    """
    Generates synthetic observations with some exceedances over threshold u.
    - p_exceed: true probability of exceedance (global) used to generate dataset
    Returns y_obs (shape N,)
    """
    rng_mu, rng_occ, rng_gpd, rng_base = random.split(rng, 4)
    var, ls, beta = 1.0, gt_ls, 1.0
    K = matern_1_2(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
    mu = dist.MultivariateNormal(jnp.zeros(s.shape[0]), K).sample(rng_mu)

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
    return y


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
    occ_mean = jnp.mean(occ_samples, axis=0).reshape(grid_size, grid_size)

    # posterior predicted excess median (set zeros where no exceed)
    excess_samples = pred["excess"]  # shape (num_samples, N) with zeros/masked where not exceed
    excess_median = jnp.median(excess_samples, axis=0).reshape(grid_size, grid_size)

    y_obs_grid = y_obs.reshape(grid_size, grid_size)
    observed_exceed = (y_obs_grid > u).astype(float)

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
