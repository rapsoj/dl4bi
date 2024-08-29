from typing import Optional

import flax.linen as nn
import jax
from jax.nn import dot_product_attention

from .mlp import MLP


class MultiheadFusedAttention(nn.Module):
    r"""Performs fused multihead query-key-value attention with dropout.

    Args:
        proj_qs: A module for projecting queries.
        proj_ks: A module for projecting keys.
        proj_vs: A module for projecting values.
        proj_out: A module for projecting output.
        num_heads: Number of heads for attention module.
        p_dropout: A dropout rate for attention.

    Returns:
        A `MultiheadFusedAttention` module.

    .. note::
        As of 2024-08-29, this requires `jax-nightly` and an NVIDIA GPU of
        series Ampere or above.

    .. note::
        This version does not support attention dropout since it is a fused
        softmax attention that operates on an optimized CUDA kernel.
    """

    proj_qs: nn.Module = MLP([64])
    proj_ks: nn.Module = MLP([64])
    proj_vs: nn.Module = MLP([64])
    proj_out: nn.Module = MLP([64])
    num_heads: int = 4
    p_dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,
        ks: jax.Array,
        vs: jax.Array,
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
        **kwargs,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of dimension $\mathbb{R}^{B\times Q\times D_{Q,K}}$
            ks: Keys of dimension $\mathbb{R}^{B\times K\times D_{Q,K}}$
            vs: Values of dimension $\mathbb{R}^{B\times K\tiems D_V}$
            valid_lens: Mask consisting of valid length per sequence of dimension
                $\mathbb{R}^B$ or $\mathbb{R}^{B\times K}$
            training: Boolean indicating whether currently training.

        Returns:
            `ctx` and `attn`, the updated values and None for `attn` weights.
        """
        qs, ks, vs = self.proj_qs(qs), self.proj_ks(ks), self.proj_vs(vs)
        (B, Q, D_QK), K, D_V, H = qs.shape, ks.shape[1], vs.shape[-1], self.num_heads
        D_QK_H, D_V_H = D_QK // H, D_V // H
        qs = self.proj_qs(qs).reshape(B, Q, H, D_QK_H)
        ks = self.proj_ks(ks).reshape(B, K, H, D_QK_H)
        vs = self.proj_vs(vs).reshape(B, K, H, D_V_H)
        ctx = dot_product_attention(
            qs, ks, vs, key_value_seq_lengths=valid_lens, implementation="cudnn"
        )
        return self.proj_out(ctx.reshape(B, Q, D_V)), None
