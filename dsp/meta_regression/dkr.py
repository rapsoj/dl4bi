from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from dsp.core.attention import MultiheadAttention

from ..core import (
    MLP,
    AddNorm,
    GaussianFourierEmbedding,
    LearnableEmbedding,
    MultiheadFastAttention,
)


class DKR(nn.Module):
    """Deep Kernel Regression.

    Args:
        depth: Number of times to apply kernel regression.
        embed_s: An embedding module for locations.
        embed_s_f: A module or combining embedded locations and function values.
        attn: An attention module.
        head: A prediction head for decoded output.

    Returns:
        An instance of the `DKR` model.
    """

    depth: int = 5
    embed_s: nn.Module = LearnableEmbedding(
        GaussianFourierEmbedding(8, 4), MLP([64, 64], nn.elu)
    )
    embed_s_f: nn.Module = MLP([64])
    # attn: nn.Module = MultiheadFastAttention()
    head: nn.Module = MLP([64] * 2 + [2])

    @nn.compact
    def __call__(
        self,
        s_ctx: jax.Array,  # [B, S_ctx, D_S]
        f_ctx: jax.Array,  # [B, S_ctx, D_F]
        s_test: jax.Array,  # [B, S_test, D_S]
        valid_lens_ctx: Optional[jax.Array] = None,  # [B]
        valid_lens_test: Optional[jax.Array] = None,  # [B]
        training: bool = False,
        **kwargs,
    ):
        r"""Run module forward.

        Args:
            rng: A psuedo-random number generator.
            s_ctx: A location array of shape `[B, S_ctx, D_S]` where
                `B` is batch size, `S_ctx` is number of context
                locations, and `D_S` is the dimension of each location.
            f_ctx: A function value array of shape `[B, S_ctx, D_F]` where `B` is
                batch size, `S_ctx` is number of context locations, and `D_F` is
                the dimension of each function value.
            s_test: A location array of shape `[B, S_test, D_S]` where `B` is
                batch size, `S_test` is number of test locations, and `D_S`
                is the dimension of each location.
            valid_lens_ctx: An optional array of shape `(B,)` indicating the
                valid positions for each `S_ctx` sequence in the batch.
            valid_lens_test: An optional array of shape `(B,)` indicating the
                valid positions for each `S_test` sequence in the batch.
            training: A boolean indicating whether this call is performed during
                training.

        Returns:
            $\mu_f,\sigma_f\in\mathbb{R}^{B\times S_\text{test}\times 2D_F}$.
        """
        add_norm = AddNorm(0.0)
        ks = self.embed_s(s_ctx, training)
        qvs = self.embed_s(s_test, training)
        kvs = self.embed_s_f(jnp.concatenate([ks, f_ctx], -1), training)
        proj_qs = MLP([64])
        proj_ks = MLP([64])
        proj_vs = MLP([64])
        for i in range(self.depth):
            attn = MultiheadAttention(proj_qs, proj_ks, proj_vs, MLP([64, 64], nn.elu))
            qvs_i, _ = attn(qvs, kvs, kvs, valid_lens_ctx)
            kvs_i, _ = attn(kvs, kvs, kvs, valid_lens_ctx)
            qvs, kvs = add_norm(qvs, qvs_i), add_norm(kvs, kvs_i)
        f_dist = self.head(qvs, training)
        f_mu, f_log_var = jnp.split(f_dist, 2, axis=-1)
        f_std = jnp.exp(f_log_var / 2)
        return f_mu, f_std
