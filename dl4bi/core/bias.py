from collections.abc import Callable
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import jit, vmap
from sps.kernels import rbf

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


# TODO(danj): support var and period?
class KernelBias(nn.Module):
    kernel: Callable = rbf
    num_heads: int = 4

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,
        ks: jax.Array,
        valid_lens_qs: Optional[jax.Array] = None,
        valid_lens_ks: Optional[jax.Array] = None,
    ):
        var = 1.0
        init_ones = nn.initializers.constant(1)
        ls = self.param("ls", init_ones, (1, self.num_heads, 1, 1))
        return vmap(rbf)(qs, ks, var, ls)
