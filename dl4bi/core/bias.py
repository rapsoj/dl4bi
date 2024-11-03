from typing import Optional

import jax
import jax.numpy as jnp
from jax import jit, vmap

from .utils import outer_subtract


@jit
def distance_bias(
    qs: jax.Array,
    ks: jax.Array,
    valid_lens_qs: Optional[jax.Array] = None,
    valid_lens_ks: Optional[jax.Array] = None,
    **kwargs,
):
    d = vmap(outer_subtract)(qs, ks)
    return -jnp.linalg.norm(d, axis=-1)[:, None, ...]  # [B, 1, Q, K]


@jit
def distance_sq_bias(
    qs: jax.Array,
    ks: jax.Array,
    valid_lens_qs: Optional[jax.Array] = None,
    valid_lens_ks: Optional[jax.Array] = None,
    **kwargs,
):
    d_sq = vmap(outer_subtract)(qs, ks) ** 2
    return -d_sq.sum(axis=-1)[:, None, ...]  # [B, 1, Q, K]
