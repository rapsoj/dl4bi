import math
from dataclasses import dataclass
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array, jit, random
from jax.lax import conv_general_dilated

from .priors import Prior
from .utils import inv_dist_sq_kernel


@dataclass
class LatticeSI:
    """A Susceptible-Infected (SI) model simulated on a lattice.

    Args:
        beta: A prior over the infection rate.
        num_init: A prior over the initial number of infected (nearest integer is used).
        kernel_width: Width of inverse distance weighted convolutional kernel
            used for transmission.

    Returns:
        An instance of the `LatticeSI` dataclass.
    """

    beta: Prior = Prior("beta", {"a": 2, "b": 18})
    num_init: Prior = Prior("uniform", {"minval": 1, "maxval": 5})
    kernel_width: int = 9

    def simulate(
        self,
        rng: Array,
        dims: Tuple[int, int] = (64, 64),
        num_steps: int = 100,
    ):
        """Simulate `num_steps` of SI model on a lattice of `dims`."""
        rng_num_init, rng = random.split(rng)
        # WARNING: this will create a separate cached jitted function
        # for each num_init possible; this is also why we don't sample
        # beta and gamma here -- each unique tuple of static args would
        # get its own cached jitted function, which can cause memory leaks
        num_init = int(self.num_init.sample(rng_num_init)[0])
        return _simulate(
            rng,
            dims,
            num_init,
            num_steps,
            self.beta,
            self.kernel_width,
        )


@partial(jit, static_argnums=tuple(range(1, 6)))
def _simulate(
    rng: jax.Array,
    dims: Tuple[int, int],
    num_init: int,
    num_steps: int,
    beta_prior: Prior,
    kernel_width: int,
):
    rng_beta, rng_gamma, rng_init, *rng_steps = random.split(rng, 3 + num_steps - 1)
    beta = beta_prior.sample(rng_beta)
    init_locs = random.choice(rng_init, math.prod(dims), (num_init,), replace=False)
    # initialize state array: 0 = susceptible, 1 = infected
    state = jnp.zeros(dims).at[jnp.unravel_index(init_locs, dims)].set(1.0)
    kernel = inv_dist_sq_kernel(kernel_width)[None, None, :, :]

    @jit
    def step(state: jax.Array, rng: jax.Array):
        neighbor_sum = conv_general_dilated(
            jnp.float32(state == 1.0)[None, None, :, :],  # infected only
            kernel,
            window_strides=(1, 1),
            padding="SAME",
        )[0, 0]  # remove batch and channel dimensions
        rng_infect, rng_recover = random.split(rng)
        u_infect = random.uniform(rng_infect, state.shape)
        new_infections = (state == 0.0) & (u_infect < beta * neighbor_sum)
        state = jnp.where(new_infections, 1.0, state)  # susceptible -> infected
        return state, state

    _, steps = jax.lax.scan(step, state, jnp.array(rng_steps))
    return jnp.vstack([state[None, ...], steps]), beta, num_init
