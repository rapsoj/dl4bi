from collections.abc import Callable
from functools import partial

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np
from cupyx.scipy.spatial import KDTree as cpKDTree
from jax import jit
from scipy.spatial import KDTree as spKDTree
from sps.kernels import l2_dist


@partial(jit, static_argnames=("k", "dist"))
def k_nearest_senders(
    r: jax.Array,
    s: jax.Array,
    k: int,
    dist: Callable = l2_dist,
):
    r"""Faster than scipy's `KDTree` for small tasks, but uses a $O(n^2)$ memory."""
    d = dist(r, s)
    idx = jnp.argsort(d, axis=-1)
    d = jnp.take_along_axis(d, idx, axis=-1)
    return idx[:, :k].flatten(), d[:, :k].flatten()


@partial(jit, static_argnames=("k",))
def scipy_k_nearest_senders(r: jax.Array, s: jax.Array, k: int):
    r"""Slower than JAX's O(n^2) implementation for small tasks, but scales in $O(N\log N)$."""
    d_shape = jax.ShapeDtypeStruct((r.shape[0], k), jnp.float32)
    idx_shape = jax.ShapeDtypeStruct((r.shape[0], k), jnp.int32)
    f = lambda r, s, k: spKDTree(np.array(s)).query(np.array(r), int(k))
    d, idx = jax.pure_callback(f, (d_shape, idx_shape), r, s, k)
    return idx.flatten(), d.flatten()


# TODO(danj): try KDTree on cuda; note this isn't yet in release 13.3.0
# https://docs.cupy.dev/en/latest/reference/generated/cupyx.scipy.spatial.KDTree.html
# https://arxiv.org/abs/2211.00120
def cupy_k_nearest_senders(r: jax.Array, s: jax.Array, k: int):
    r = cp.asarray(r.addressable_data(0))
    s = cp.asarray(s.addressable_data(0))
    d, idx = cpKDTree(s).query(r, k)
    return jnp.array(idx).flatten(), jnp.array(d).flatten()
