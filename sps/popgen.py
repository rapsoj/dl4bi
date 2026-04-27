from dataclasses import dataclass, replace
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array, jit, lax, random
from jax.lax import conv_general_dilated

from .priors import Prior


@dataclass(frozen=True)
class PopGenState:
    migration: Array  # [1]
    mutation: Array  # [1]
    population: Array  # [1]
    prevalence: Array  # [B, C, H, W]


jax.tree_util.register_pytree_node(
    PopGenState,
    lambda d: ((d.migration, d.mutation, d.population, d.prevalence), None),
    lambda _aux, children: PopGenState(*children),
)


@dataclass
class PopGen:
    """A Population Genetics simulator on a lattice.

    Args:
        migration: A prior over the migration rate.
        mutation: A prior over the mutation rate.
        population: A prior over the population size per deme.

    Returns:
        An instance of the `PopGen` dataclass.
    """

    migration: Prior = Prior("uniform", {"minval": 10**-3.3, "maxval": 10**-1.3})
    mutation: Prior = Prior("uniform", {"minval": 10e-6, "maxval": 1e-3})
    population: Prior = Prior("fixed", {"value": 1000})

    def simulate(
        self,
        rng: Array,
        num_warmup: int = 2000,
        num_steps: int = 16,
        step_interval: int = 16,
        batch_size: int = 32,
        dims: Tuple[int, int] = (32, 32),
        wrap_edges: bool = True,
        state: Optional[PopGenState] = None,
    ):
        """
        Args:
            rng: Random number key.
            num_warmup: Number of warmup steps (thrown away).
            num_steps: Total number of steps kept at the end.
            step_interval: Number of steps to skip between kept steps.
            batch_size: Number of sequences of steps to keep.
            dims: Surface deme dimensions, HxW.
            wrap_edges: Connect the top and bottom and left and right
                sides of the surface.
            state: Continue from this `PopGenState`.
        """
        if state is None:
            rng_mi, rng_mu, rng_po, rng = random.split(rng, 4)
            migration = self.migration.sample(rng_mi, (1,))
            mutation = self.mutation.sample(rng_mu, (1,))
            population = self.population.sample(rng_po, (1,))
            prevalence = jnp.zeros((batch_size, 1, *dims))  # [B, C=1, H, W]
            state = PopGenState(migration, mutation, population, prevalence)
        return _simulate(
            rng,
            state,
            num_warmup,
            num_steps,
            step_interval,
            wrap_edges,
        )


@partial(
    jit,
    static_argnames=(
        "num_warmup",
        "num_steps",
        "step_interval",
        "wrap_edges",
    ),
)
def _simulate(
    rng: Array,
    state: PopGenState,
    num_warmup: int,
    num_steps: int,
    step_interval: int,
    wrap_edges: bool = True,
):
    """
    Simulate evolution under a lattice model assuming two alleles.
    """
    migration = state.migration
    mutation = state.mutation
    population = state.population
    prevalence = state.prevalence
    T, (B, C, H, W) = num_steps, prevalence.shape
    buffer = jnp.zeros((T, B, C, H, W))

    def step(carry, i):
        rng, buffer, prevalence = carry
        rng, rng_step = random.split(rng)
        prevalence = _migrate_and_mutate(migration, mutation, prevalence, wrap_edges)
        prevalence = random.binomial(rng_step, population, prevalence) / population
        idx = (i - num_warmup) // step_interval
        update = lambda b: b.at[idx].set(prevalence)
        keep = jnp.logical_and(i >= num_warmup, (i - num_warmup) % step_interval == 0)
        buffer = lax.cond(keep, update, lambda b: b, buffer)
        return (rng, buffer, prevalence), None

    total_steps = num_warmup + num_steps * step_interval
    (rng, buffer, last_prev), _ = lax.scan(
        step, (rng, buffer, prevalence), jnp.arange(total_steps)
    )
    prevalences = jnp.moveaxis(buffer, 0, 2)  # [B, C, T, H, W]
    last_state = replace(state, prevalence=last_prev)
    return prevalences, last_state


@partial(jit, static_argnames=("wrap_edges"))
def _migrate_and_mutate(
    migration: Array,  # [1]
    mutation: Array,  # [1]
    prevalence: Array,  # [B, C, H, W]
    wrap_edges: bool = True,
):
    """
    Compute migration and mutation.

    .. note::
        `wrap_edges` specifies that the left hand side wraps over to the
        right, and same with top and bottom of the image. If `False`, this uses
        reflective boundary conditions, which means each edge is padded with a
        mirrored version of the data.
    """
    k_neighbor = jnp.array(
        [
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.0],
        ]
    )
    k_center = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    kernel = k_neighbor * migration + k_center * (1 - 2 * migration)
    prevalence_padded = jnp.pad(
        prevalence,
        ((0, 0), (0, 0), (1, 1), (1, 1)),
        mode="wrap" if wrap_edges else "edge",
    )
    prevalence_migrated = conv_general_dilated(
        lhs=prevalence_padded,
        rhs=kernel[None, None, :, :],
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return prevalence_migrated * (1 - mutation) + 0.5 * mutation
