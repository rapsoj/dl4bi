from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


class DotScorer(nn.Module):
    r"""Performs dot product attention scoring.

    $$a(\mathbf{q},\mathbf{k}) = \frac{\mathbf{q}^\intercal\mathbf{k}}{\sqrt{d}}$$
    """

    @nn.compact
    def __call__(self, qs: jax.Array, ks: jax.Array):
        d = qs.shape[-1]
        return jnp.einsum("bqd,bkd->bqk", qs, ks) / jnp.sqrt(d)


class AdditiveScorer(nn.Module):
    r"""Performs additive attention scoring.

    $$a(\mathbf{q},\mathbf{k}) = \mathbf{w}_v^\intercal(\mathbf{W}_q\mathbf{q}+\mathbf{W}_k\mathbf{k})$$
    """

    num_hidden: int = 128

    @nn.compact
    def __call__(self, qs: jax.Array, ks: jax.Array):
        qs = nn.Dense(self.num_hidden, use_bias=False)(qs)
        ks = nn.Dense(self.num_hidden, use_bias=False)(ks)
        # [B, Q, 1, H] + [B, 1, K, H]
        feats = jnp.expand_dims(qs, axis=2) + jnp.expand_dims(ks, axis=1)
        feats = nn.tanh(feats)
        return nn.Dense(1, use_bias=False)(feats).squeeze(-1)


class Attention(nn.Module):
    """Performs (masked) query-key-value attention with dropout."""

    scorer: nn.Module = DotScorer()
    p_dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, Q, D_Q]
        ks: jax.Array,  # [B, K, D_K]
        vs: jax.Array,  # [B, V, D_V]
        valid_lens: Optional[jax.Array] = None,  # [B] or [B, Q]
        training=False,
    ):
        d = qs.shape[-1]
        scores = self.scorer(qs, ks)
        attn = masked_softmax(scores, valid_lens)
        attn = nn.Dropout(self.p_dropout, deterministic=not training)(attn)
        return attn @ vs, attn


# source: https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html
def masked_softmax(scores: jax.Array, valid_lens: Optional[jax.Array] = None):
    """Performs softmax on a 3D logits array using an optional 1 or 2 dim `valid_lens`."""
    if valid_lens is None:
        return nn.softmax(scores, axis=-1)

    def _sequence_mask(logits: jax.Array, valid_len: jax.Array):
        max_len = logits.shape[1]
        mask = jnp.arange(max_len, dtype=jnp.float32)[None, :] < valid_len[:, None]
        return jnp.where(mask, logits, -1e6)

    B, Q, K = scores.shape
    if valid_lens.ndim == 1:
        valid_lens = jnp.repeat(valid_lens, Q)
    else:
        valid_lens = valid_lens.reshape(-1)
    logits = _sequence_mask(scores.reshape(-1, K), valid_lens)
    return nn.softmax(logits.reshape((B, Q, K)), axis=-1)
