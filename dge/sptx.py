from collections.abc import Callable
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from .attention import Attention
from .embed import LearnableEmbedding
from .mlp import MLP
from .transformer import AddNorm


class KRBlock(nn.Module):
    """A Kernel Regression Block.

    Args:
        attn: An attention module.
        p_dropout: Dropout rate `AddNorm`s.
        d_ffn: Optional dim for feed forward, defaults to twice the last
            dimension of input `kvs`.
        act_fn: Activation function, defaults to relu.

    Returns:
        An instance of the `KRBlock` model.
    """

    attn: nn.Module = Attention()
    p_dropout: float = 0.0
    d_ffn: Optional[int] = None
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(
        self,
        qvs: jax.Array,
        kvs: jax.Array,
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
    ):
        d = kvs.shape[-1]
        d_ffn = self.d_ffn or 2 * d
        add_norm = AddNorm(self.p_dropout)
        ffn = nn.Sequential([nn.Dense(d_ffn), self.act_fn, nn.Dense(d)])
        qvs2, _ = self.attn(qvs, kvs, kvs, valid_lens)
        kvs2, _ = self.attn(kvs, kvs, kvs, valid_lens)
        qvs3, kvs3 = add_norm(qvs, qvs2), add_norm(kvs, kvs2)
        qvs4, kvs4 = ffn(qvs3), ffn(kvs3)
        return add_norm(qvs3, qvs4), add_norm(kvs3, kvs4)


class KRStack(nn.Module):
    """A stack of `KRBlock`s.

    Args:
        attn: An attention module.
        num_blks: Number of `KRBlock`s.
        p_dropout: Dropout rate `AddNorm`s.
        d_ffn: Optional dim for feed forward, defaults to twice the last
            dimension of input `kvs`.

    Returns:
        An instnace of a `KRStack`.
    """

    attn: nn.Module = Attention()
    num_blks: int = 3
    p_dropout: float = 0.0
    d_ffn: Optional[int] = None
    act_fn: Callable = nn.relu
    skip_every_n: int = 3

    @nn.compact
    def __call__(
        self,
        qvs: jax.Array,
        kvs: jax.Array,
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
    ):
        d_ffn = self.d_ffn or 2 * kvs.shape[-1]
        add_norm = AddNorm(self.p_dropout)
        qvs, kvs = KRBlock(
            self.attn,
            self.p_dropout,
            d_ffn,
            self.act_fn,
        )(qvs, kvs, valid_lens, training)
        skip_qvs, skip_kvs = qvs, kvs
        for i in range(1, self.num_blks):
            is_skip = i % self.skip_every_n == 0
            if is_skip:
                qvs = add_norm(skip_qvs, qvs)
                kvs = add_norm(skip_kvs, kvs)
            qvs, kvs = KRBlock(
                self.attn.copy(name=f"attn_{i}"),
                self.p_dropout,
                d_ffn,
                self.act_fn,
            )(qvs, kvs, valid_lens, training)
            if is_skip:
                skip_qvs = qvs
                skip_kvs = kvs
        return qvs, kvs


class SPTx(nn.Module):
    """A Stochastic Process Transformer (SPTx).

    Args:
        embed_s: An embedding module for locations.
        embed_s_f: A module or combining embedded locations and function values.
        dec: A decoder module, e.g. a `KRStack`.
        head: A prediction head for decoded output.

    Returns:
        An instance of the `SPTx` model.
    """

    embed_s: nn.Module = LearnableEmbedding(lambda x: x, MLP([128] * 3))
    embed_s_f: nn.Module = MLP([128])
    dec: nn.Module = KRStack()
    head: nn.Module = MLP([128] * 2 + [2])

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
            $\mu_f,\log(\sigma_f^2\in\mathbb{R}^{B\times S_\text{test}\times 2D_F}$.
        """
        s_ctx_embed = self.embed_s(s_ctx, training)
        s_test_embed = self.embed_s(s_test, training)
        s_f_ctx = jnp.concatenate([s_ctx_embed, f_ctx], -1)
        s_f_ctx_embed = self.embed_s_f(s_f_ctx, training)
        s_f_test_enc, _ = self.dec(
            s_test_embed,
            s_f_ctx_embed,
            valid_lens_ctx,
            training,
        )
        f_dist = self.head(s_f_test_enc, training)
        f_mu, f_log_var = jnp.split(f_dist, 2, axis=-1)
        return f_mu, f_log_var
