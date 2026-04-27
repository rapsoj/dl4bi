from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, random
from jax.tree_util import Partial


@dataclass
class Prior:
    """Represents a prior using `jax.random` distributions.

    Args:
        dist: A distribution name from `jax.random` or `kernels` submodule.
        kwargs: A dict of the parameters for the given distribution.

    Returns:
        An instance of the Prior dataclass.
    """

    dist: str
    kwargs: dict[str, float]

    def __post_init__(self):
        dist_func = globals().get(self.dist, getattr(random, self.dist, None))
        self.dist_func = Partial(dist_func, **self.kwargs)

    def __hash__(self):
        return hash((self.dist, repr(self.kwargs)))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def sample(self, rng: Array, shape: Sequence[int] = (1,)) -> Array:
        """Samples this prior.

        Args:
            rng: A psuedo-random number generator from `jax.random`.
            shape: Output shape of sample(s).

        Returns:
            A sample of shape `shape`.
        """
        return self.dist_func(rng, shape=shape)


# JAX doesn't have a parameterized normal
# https://jax.readthedocs.io/en/latest/_autosummary/jax.random.normal.html
def normal(
    rng: Array,
    mu: float,
    sigma: float,
    shape: Sequence[int],
) -> Array:
    """Normal distribution parameterized by `mu` and `sigma`.

    Args:
        rng: A psuedo-random number generator from `jax.random`.
        mu: Location, or center of distribution.
        sigma: Standard deviation.
        shape: Output shape of sample(s).

    Returns:
        A sample of shape `shape`.
    """
    return mu + sigma * random.normal(rng, shape)


# JAX doesn't have a lambda parameterized exponential
# https://jax.readthedocs.io/en/latest/_autosummary/jax.random.exponential.html
def exponential(
    rng: Array,
    lam: float,
    shape: Sequence[int],
) -> Array:
    """Exponential parameterized by lambda `lam`.

    Args:
        rng: A psuedo-random number generator from `jax.random`.
        lam: Lambda of exponential distribution.
        shape: Output shape of sample(s).

    Returns:
        A sample of shape `shape`.
    """
    return 1 / lam * random.exponential(rng, shape)


# JAX doesn't have a rate parameterized gamma
# https://jax.readthedocs.io/en/latest/_autosummary/jax.random.gamma.html
def gamma(
    rng: Array,
    alpha: float,  # shape
    beta: float,  # rate
    shape: Sequence[int],
) -> Array:
    """Gamma parameterized by `alpha` (shape) and `beta` (rate).

    Args:
        rng: A psuedo-random number generator from `jax.random`.
        alpha: The standard shape parameters.
        beta: The rate parameter.
        shape: Output shape of sample(s).

    Returns:
        A sample of shape `shape`.
    """
    return random.gamma(rng, alpha, shape) / beta


# JAX doesn't have a lambda parameterized inverse-gamma
# https://jax.readthedocs.io/en/latest/_autosummary/jax.random.gamma.html
def inverse_gamma(
    rng: Array,
    alpha: float,
    beta: float,
    shape: Sequence[int],
) -> Array:
    """Inverse-Gamma parameterized by `alpha` (shape) and `beta` (rate).

    Args:
        rng: A psuedo-random number generator from `jax.random`.
        alpha: The standard shape parameters.
        beta: The rate parameter.
        shape: Output shape of sample(s).

    Returns:
        A sample of shape `shape`.
    """
    return 1 / gamma(rng, alpha, beta, shape)


def fixed(
    rng: Array,
    value: float,
    shape: Sequence[int],
) -> Array:
    """Fixed distribution.

    Args:
        rng: A psuedo-random number generator from `jax.random`.
        value: A fixed value to return for all samples.
        shape: Output shape of sample(s).

    Returns:
        A fixed sample of shape `shape`.
    """
    return jnp.full(shape, value)
