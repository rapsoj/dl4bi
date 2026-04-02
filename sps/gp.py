from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import jax.numpy as jnp
from jax import Array, enable_x64, jit, lax, random, vmap
from jax.tree_util import Partial
from jax.typing import ArrayLike

from .kernels import matern_3_2
from .priors import Prior


@dataclass
class GP:
    """Gaussian Process simulator.

    Args:
        kernel: A kernel from the `kernels` submodule.
        var: The variance prior. Distributions include those in
            `jax.random` as well as those in the `priors` submodule.
        ls: The lengthscale prior. Distributions include those in
            `jax.random` as well as those in the `priors` submodule.
        period: Used only for periodic kernels.
        jitter: Jitter added to diagonal of covariance matrix to numerically
            stabilize decomposition.

    Returns:
        An instance of the `GP` dataclass.
    """

    kernel: Callable = matern_3_2
    var: Prior = Prior("fixed", {"value": 1})
    ls: Prior = Prior("beta", {"a": 2.5, "b": 6.0})
    period: Optional[Prior] = None
    jitter: float = 1e-5

    def simulate(
        self,
        rng: Array,
        locations: ArrayLike,  # [..., D]
        batch_size: int = 1,
        approx: bool = False,
    ) -> tuple[Array, Array, Array, None | Array, Array]:
        r"""Simulate `batch_size` realizations of the GP at `locations`.

        Each batch is sampled as follows:

        $
        \begin{aligned}
        \text{var}&\sim p_\text{var}(\cdot) \\\\
        \text{ls}&\sim p_\text{ls}(\cdot) \\\\
        \mathbf{z}&\sim \mathcal{N}(0, 1, \text{shape}=(\text{batch\_size}, \text{num\_locations})) \\\\
        \mathbf{K}&=\text{kernel}(\text{locations},\text{locations},\text{var},\text{ls}) \\\\
        \mathbf{L}&=\text{Cholesky}(\mathbf{K}) \\\\
        \mathbf{f}&=\mathbf{Lz} \\\\
        \end{aligned}
        $

        This sampling process is designed to amortize the cost of
        the Cholesky decomposition, which is $\mathcal{O}(n^3)$,
        by sampling `(batch_size, num_locations)` `z`s for each `L`.

        Args:
            rng: A psuedo-random number generator from `jax.random`.
            locations: An array of locations where the last dimension
                is the dimension of single location, i.e. if a location
                is 3 dimensional, the last dimension of the array is 3.
            batch_size: The number of samples to generate.
            approx: Approximate samples using Kronecker factorization,
                otherwise use "exact" Cholesky decomposition.

        Returns:
            `f`, `var`, `ls`, `period`, and `z`.
        """
        locations = locations[:, None] if locations.ndim == 1 else locations
        rng_var, rng_ls, rng_period, rng_z = random.split(rng, 4)
        num_locations = locations.size // locations.shape[-1]
        var = self.var.sample(rng_var)
        ls = self.ls.sample(rng_ls)
        z = random.normal(rng_z, shape=(batch_size, num_locations))
        kernel = self.kernel
        period = None
        if self.period is not None:
            period = self.period.sample(rng_period)
            kernel = Partial(self.kernel, period=period)
        sample = kronecker if approx else cholesky
        with enable_x64():  # TODO(danj): make this optional
            f = sample(kernel, jnp.float64(locations), var, ls, z, self.jitter)
            f = jnp.float32(f)
        f = f.reshape(-1, *locations.shape[:-1], 1)  # batch x grid x 1
        return f, var, ls, period, z


def cholesky(
    kernel: Callable,
    locations: ArrayLike,  # [..., D]
    var: Array,
    ls: Array,
    z: Array,  # [B, L]
    jitter: float = 1e-5,
) -> Array:
    """Creates samples using Cholesky covariance factorization.

    Args:
        kernel: A kernel function.
        locations: An array of locations where the last dimension
            is the dimension of single location, i.e. if a location
            is 3 dimensional, the last dimension of the array is 3.
        var: The variance parameter.
        ls: The lengthscale parameter.
        z: A random vector used to generate samples.
        noise: Noise added for numerical stability in Cholesky
            decomposition. Insufficiently large values will result
            in nan values.

    Returns:
        `Lz`: samples from the kernel combined with a random vector `z`.
    """
    num_locations = locations.size // locations.shape[-1]
    K = kernel(locations, locations, var, ls) + jitter * jnp.eye(num_locations)
    L = lax.linalg.cholesky(K)
    return jnp.einsum("ij,bj->bi", L, z)


def kronecker(
    kernel: Callable,
    locations: ArrayLike,  # [..., D]
    var: Array,
    ls: Array,
    z: Array,  # [B, L]
    jitter: float = 1e-5,
) -> Array:
    """Creates samples using Kronecker covariance factorization.

    Source: https://proceedings.mlr.press/v37/flaxman15.html

    Args:
        kernel: A kernel function.
        locations: An array of locations where the last dimension
            is the dimension of single location, i.e. if a location
            is 3 dimensional, the last dimension of the array is 3.
        var: The variance parameter.
        ls: The lengthscale parameter.
        z: A random vector used to generate samples.
        noise: Noise added for numerical stability in Cholesky
            decomposition. Insufficiently large values will result
            in nan values.

    Returns:
        `Lz`: samples from the kernel combined with a random vector `z`.
    """
    Ls = _kronecker_Ls(kernel, locations, var, ls, jitter)
    return vmap(_kronecker_mvprod, in_axes=(None, 0))(Ls, z)


def _kronecker_Ls(
    kernel: Callable,
    locations: ArrayLike,
    var: float,
    ls: float,
    jitter: float = 1e-5,
) -> Sequence[Array]:
    """Calculates Cholesky decomposition of each dimension in covariance matrix.

    Args:
        z: Random vector to be multiplied by combined L.

    Returns:
        `Ls`: Cholesky decompositions of each dimension in covariance matrix.
    """
    D, Ls = locations.shape[-1], []
    start, stop = jnp.zeros(D, dtype=int), jnp.ones(D, dtype=int)
    for dim, dim_size in enumerate(locations.shape[:-1]):
        _stop = stop.at[dim].set(dim_size)
        axis = lax.slice(locations[..., dim], start, _stop).squeeze()[..., jnp.newaxis]
        K = kernel(axis, axis, var, ls) + jitter * jnp.eye(dim_size)
        Ls += [jnp.linalg.cholesky(K)]
    return Ls


@jit
def _kronecker_mvprod(Ls: Sequence[Array], z: Array) -> Array:
    """Linear Kronecker product of `Ls` with vector `z`.

    Source: https://mlg.eng.cam.ac.uk/pub/pdf/Saa11.pdf p137

    Args:
        Ls: Cholesky decompositions of each dimension.
        z: Random vector to be multiplied by combined L.

    Returns:
        `Lz`: samples from the kernel combined with random vector `z`.
    """
    x, N = z, z.size
    for L in Ls:
        D = L.shape[0]
        X = x.reshape(D, N // D)
        Z = L @ X
        x = Z.T.flatten()
    return x
