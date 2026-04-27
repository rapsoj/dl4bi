from dataclasses import dataclass
from typing import Dict, Callable
import jax.numpy as jnp
from jax import random

from sps.kernels import (
    st_separable_rbf_matern_1_2,
    st_nonseparable_gneiting,
    st_nonseparable_gneiting_advected,
)


@dataclass
class KernelFamily:
    name: str
    compute_fn: Callable
    param_sampler: Callable
    cond_builder: Callable

    def sample_params(self, rng) -> Dict:
        return self.param_sampler(rng)

    def compute(self, x, y, params: Dict):
        return self.compute_fn(x, y, **params)

    def build_conditionals(self, params: Dict):
        return self.cond_builder(params)


# --- unified conditional builder ---
def build_full_cond(params):
    return jnp.array([
        params.get("ls_space", 0.0),
        params.get("a", 0.0),
        params.get("alpha", 0.0),
        params.get("beta", 0.0),
        *(params.get("v", jnp.zeros(2)))
    ])


# --- separable ---
def separable_param_sampler(rng):
    return {
        "var": 1.0,
        "ls_space": random.uniform(rng, (), minval=5.0, maxval=50.0),
        "ls_time": 1.0,
    }


separable_kernel_family = KernelFamily(
    name="separable",
    compute_fn=st_separable_rbf_matern_1_2,
    param_sampler=separable_param_sampler,
    cond_builder=build_full_cond,
)


# --- non-separable ---
def nonsep_param_sampler(rng):
    rng_ls, rng_a, rng_alpha, rng_beta = random.split(rng, 4)
    return {
        "var": 1.0,
        "ls_space": random.uniform(rng_ls, (), minval=5.0, maxval=50.0),
        "a": random.uniform(rng_a, (), minval=0.1, maxval=2.0),
        "alpha": random.uniform(rng_alpha, (), minval=0.3, maxval=1.0),
        "beta": random.uniform(rng_beta, (), minval=0.0, maxval=1.0),
    }


nonsep_kernel_family = KernelFamily(
    name="nonseparable",
    compute_fn=st_nonseparable_gneiting,
    param_sampler=nonsep_param_sampler,
    cond_builder=build_full_cond,
)


# --- advected ---
def advected_param_sampler(rng):
    rng_ls, rng_a, rng_alpha, rng_beta, rng_v = random.split(rng, 5)
    return {
        "var": 1.0,
        "ls_space": random.uniform(rng_ls, (), minval=5.0, maxval=50.0),
        "a": random.uniform(rng_a, (), minval=0.1, maxval=2.0),
        "alpha": random.uniform(rng_alpha, (), minval=0.3, maxval=1.0),
        "beta": random.uniform(rng_beta, (), minval=0.0, maxval=1.0),
        "v": random.normal(rng_v, (2,)) * 0.5,
    }


advected_kernel_family = KernelFamily(
    name="advected",
    compute_fn=st_nonseparable_gneiting_advected,
    param_sampler=advected_param_sampler,
    cond_builder=build_full_cond,
)