#!/usr/bin/env python
import argparse
import sys
from typing import Optional

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive, log_likelihood
from sps.kernels import rbf


# source: https://num.pyro.ai/en/stable/examples/gp.html
def simple_gp_model(s: jax.Array, f: Optional[jax.Array] = None):
    var = numpyro.sample("var", dist.LogNormal(0.0, 1.0))
    ls = numpyro.sample("ls", dist.LogNormal(0.0, 1.0))
    k = rbf(s, s, var, ls) + 1e-6 * jnp.eye(s.shape[0])
    numpyro.sample(
        "f",
        dist.MultivariateNormal(jnp.zeros(s.shape[0]), covariance_matrix=k),
        obs=f,
    )


def infer(rng, args, model, s, f):
    sampler = NUTS(model)
    mcmc = MCMC(
        sampler,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
    )
    mcmc.run(rng, s, f)
    return mcmc


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-s", "--seed", default=0, type=int)
    parser.add_argument("-nw", "--num_warmup", default=1000, type=int)
    parser.add_argument("-ns", "--num_samples", default=1000, type=int)
    parser.add_argument("-nc", "--num_chains", default=1, type=int)
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    rng_prior_pred, rng_mcmc = random.split(random.key(args.seed))
    s = jnp.linspace(-2, 2, 128)
    prior_pred = Predictive(simple_gp_model, num_samples=1)
    prior_samples = prior_pred(rng_prior_pred, s=s)
    latents = set(prior_samples) - {"f"}
    print({f"{k}: {v[0]:0.4f}" for k, v in prior_samples.items() if k in latents})
    f = prior_samples["f"][0]
    mcmc = infer(rng_mcmc, args, simple_gp_model, s, f)
    mcmc.print_summary()
    posterior_samples = mcmc.get_samples()
    ll = log_likelihood(simple_gp_model, posterior_samples, s=s, f=f)
    print(ll)
