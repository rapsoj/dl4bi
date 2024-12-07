from collections.abc import Callable
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
from jax import jit
from jraph import GraphsTuple
from scipy.spatial import KDTree
from sps.kernels import l2_dist

from ..core import MLP, GraphKRBlock, mask_from_valid_lens


def custom_k_nearest_senders(
    rx: jax.Array,
    tx: jax.Array,
    k: int,
    dist: Callable = l2_dist,
):
    """Retrieves k-nearest senders, but uses a $O(n^2)$ memory."""
    d = dist(rx, tx)
    idx = jnp.argsort(d, axis=-1)
    d = jnp.take_along_axis(d, idx, axis=-1)
    return idx[:, :k].flatten(), d[:, :k].flatten()


# TODO(danj): use jax.pure_callback to make this jit-compatible
def k_nearest_senders(rx: jax.Array, tx: jax.Array, k: int):
    d, idx = KDTree(tx).query(rx, k)
    return idx.flatten(), d.flatten()


# TODO(danj): try KDTree on cuda
# https://docs.cupy.dev/en/latest/reference/generated/cupyx.scipy.spatial.KDTree.html
# https://arxiv.org/abs/2211.00120
def k_nearest_senders_gpu(rx: jax.Array, tx: jax.Array, k: int):
    raise NotImplementedError()


# TODO(danj): include global vnode conditioning
class DSKR(nn.Module):
    """DSKR

    .. note::
        Fixed effects can be embedded with `embed_s`, i.e. if the "index"
        consists of [fixed effects, space, time], `embed_s` could be a Flax
        module that embeds fixed effects, space, and time separately and
        concatenates the output.

    .. note::
        When the index set, `s`, includes fixed effects or features that
        do not factor into calculating the k-nearest neighbors, you
        can override `k_nearest_senders`.

    .. warning::
        `min(valid_lens_ctx)` and `min(valid_lens_test)` must both
        be greater than `k`.
    """

    k: int = 10
    k_nearest_senders: Callable = custom_k_nearest_senders
    num_blks: int = 6
    num_reps: int = 1
    min_std: float = 0.0
    embed_s: Callable = lambda x: x
    embed_f: Callable = lambda x: x
    embed_obs: nn.Module = nn.Embed(2, 4)
    embed_all: nn.Module = MLP([256, 128, 64], nn.gelu)
    blk: nn.Module = GraphKRBlock()
    norm: nn.Module = nn.LayerNorm()
    head: nn.Module = MLP([256, 64, 2], nn.gelu)

    @nn.compact
    def __call__(
        self,
        s_ctx: jax.Array,  # [B, L_ctx, S]
        f_ctx: jax.Array,  # [B, L_ctx, F]
        s_test: jax.Array,  # [B, L_test, S]
        valid_lens_ctx: Optional[jax.Array] = None,  # [B]
        valid_lens_test: Optional[jax.Array] = None,  # [B]
        training: bool = False,
        **kwargs,
    ):
        (B, N_t), N_c = s_test.shape[:-1], s_ctx.shape[1]
        if valid_lens_ctx is None:
            valid_lens_ctx = jnp.repeat(N_c, B)
        # construct node features
        stack = lambda *args: jnp.concatenate(args, axis=-1)
        f_test = jnp.zeros([*s_test.shape[:-1], f_ctx.shape[-1]])
        obs = jnp.ones(f_ctx.shape[:-1], dtype=jnp.uint8)
        unobs = jnp.zeros(f_test.shape[:-1], dtype=jnp.uint8)
        ctx = stack(self.embed_obs(obs), self.embed_s(s_ctx), self.embed_f(f_ctx))
        test = stack(self.embed_obs(unobs), self.embed_s(s_test), self.embed_f(f_test))
        x_ctx, x_test = self.norm(self.embed_all(ctx)), self.norm(self.embed_all(test))
        # build localized graphs
        graphs = []
        receivers = lambda n: jnp.repeat(jnp.arange(n), self.k)
        mask = mask_from_valid_lens(N_c, valid_lens_ctx)
        s_send = jnp.where(mask, s_ctx, 1e6)
        # TODO(danj): unloop this (vmap)
        # TODO(danj): remove graphs altogether...
        for b in range(B):
            senders_ctx, d_ctx = self.k_nearest_senders(s_ctx[b], s_send[b], self.k)
            senders_test, d_test = self.k_nearest_senders(s_test[b], s_send[b], self.k)
            g = GraphsTuple(
                nodes=jnp.vstack([x_ctx[b], x_test[b]]),
                edges=stack(d_ctx, d_test),
                senders=stack(senders_ctx, senders_test),
                receivers=stack(receivers(N_c), N_c + receivers(N_t)),
                n_node=jnp.array([N_c + N_t]),
                n_edge=jnp.array([(N_c + N_t) * self.k]),
                globals=jnp.max(x_ctx, axis=0, keepdims=True),  # TODO(danj): attn?
            )
            graphs += [g]
        graphs = jraph.batch(graphs)
        for _ in range(self.num_blks):
            blk = self.blk.copy()
            for _ in range(self.num_reps):
                graphs = blk(graphs, training)
        x_t = jnp.stack([g.nodes[-N_t:] for g in jraph.unbatch(graphs)])
        f_dist = self.head(x_t, training)
        f_mu, f_log_var = jnp.split(f_dist, 2, axis=-1)
        f_std = jnp.exp(f_log_var / 2)
        f_std = self.min_std + (1 - self.min_std) * f_std
        return f_mu, f_std
