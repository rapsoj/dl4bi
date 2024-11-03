from collections.abc import Callable
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import jit, vmap
from sps.kernels import rbf

from .mlp import MLP
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
    # [B, 1, Q, K]
    return (vmap(outer_subtract)(qs, ks) ** 2).sum(axis=-1)[:, None, ...]


# TODO(danj): support multiple heads; support var and period?
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
        init_ones = nn.initializers.constant(1)
        var, ls = 1.0, self.param("ls", init_ones, (1,))
        return vmap(rbf, in_axes=(0, 0, None, None))(qs, ks, var, ls)[:, None, ...]


# TODO(danj): support multiple heads
class EmbedDistanceBias(nn.Module):
    embed: nn.Module

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,
        ks: jax.Array,
        valid_lens_qs: Optional[jax.Array] = None,
        valid_lens_ks: Optional[jax.Array] = None,
    ):
        (B, Q, _), K = qs.shape, ks.shape[1]
        d_sq = distance_sq_bias(qs, ks, valid_lens_qs, valid_lens_ks)[..., None]
        return self.embed(d_sq).reshape(B, 1, Q, K)
