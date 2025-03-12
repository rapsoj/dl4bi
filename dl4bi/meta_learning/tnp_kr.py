from collections.abc import Callable
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import jit, vmap

from ..core.attention import MultiHeadAttention, RBFNetworkBiasedScanAttention
from ..core.mlp import MLP
from ..core.transformer import KRBlock
from .model_output import DiagonalMVNOutput
from .steps import likelihood_train_step, likelihood_valid_step


class TNPKR(nn.Module):
    """Transformer Neural Process - Kernel Regression (TNP-KR).

    Args:
        num_blks: Number of `KRBlocks` to use.
        num_reps: Number of times to repeat each `KRBlock`.
        embed_s: A module that embeds the index set prior to embedding with
            function values.
        embed_f: A module that embeds function values prior to embedding with
            the index set.
        embed_obs: A module that creates embeddings for observed (context) and
            unobserved (test) points.
        embed_all: A module that jointly embeds `obs`, `s`, and `f` embeddings.
        x_bias: Bias module for fixed effects.
        s_bias: Bias module for spatial effects.
        t_bias: Bias module for temporal effects.
        blk: A block to use for each layer and each repetition.
        norm: A module used for normalization between blocks.
        head: Transforms the tokens into model output.
        output_fn: A function that transforms the model output into
            a form that can be consumed by loss functions.
        train_step: What training step to use.
        valid_step: What validation step to use.

    Returns:
        An instance of the `TNP-KR` model.
    """

    num_blks: int = 6
    num_reps: int = 1
    embed_s: Callable = lambda x: x
    embed_f: Callable = lambda x: x
    embed_obs: nn.Module = nn.Embed(2, 4)
    embed_all: nn.Module = MLP([256, 128, 64], nn.gelu)
    x_bias: Optional[Callable] = None
    s_bias: Optional[Callable] = None
    t_bias: Optional[Callable] = None
    blk: nn.Module = KRBlock()
    norm: nn.Module = nn.LayerNorm()
    head: nn.Module = MLP([256, 64, 2], nn.gelu)
    output_fn: Callable = DiagonalMVNOutput.from_conditional_np
    train_step: Callable = likelihood_train_step
    valid_step: Callable = likelihood_valid_step

    @nn.compact
    def __call__(
        self,
        x_ctx: Optional[jax.Array] = None,  # [B, L_ctx, D_x]
        s_ctx: Optional[jax.Array] = None,  # [B, L_ctx, D_s]
        t_ctx: Optional[jax.Array] = None,  # [B, L_ctx, D_s]
        f_ctx: Optional[jax.Array] = None,  # [B, L_ctx, D_f]
        mask_ctx: Optional[jax.Array] = None,  # [B, L_ctx]
        x_test: Optional[jax.Array] = None,  # [B, L_test, D_x]
        s_test: Optional[jax.Array] = None,  # [B, L_test, D_s]
        t_test: Optional[jax.Array] = None,  # [B, L_test, D_t]
        training: bool = False,
        **kwargs,
    ):
        r"""Run module forward.

        Args:
            rng: A psuedo-random number generator.
            x_ctx: Optional fixed effects for context points.
            t_ctx: Optional temporal values for context points.
            s_ctx: Optional spatial values for context points.
            f_ctx: Function values for context points.
            mask_ctx: A mask for context points.
            x_test: Optional fixed effects for test points.
            t_test: Optional temporal values for test points.
            s_test: Optional spatial values for test points.
            f_test: Function values for test points.
            training: A boolean indicating whether this call is performed during
                training.

        Returns:
            `ModelOutput`.
        """
        norm = nn.LayerNorm()
        stack = lambda *args: jnp.concatenate([x for x in args if x.size > 0], axis=-1)
        f_test = jnp.zeros([*s_test.shape[:-1], f_ctx.shape[-1]])
        obs = jnp.ones(f_ctx.shape[:-1], dtype=jnp.uint8)
        unobs = jnp.zeros(f_test.shape[:-1], dtype=jnp.uint8)
        ctx = stack(self.embed_obs(obs), self.embed_s(s_ctx), self.embed_f(f_ctx))
        test = stack(self.embed_obs(unobs), self.embed_s(s_test), self.embed_f(f_test))
        qvs, kvs = norm(self.embed_all(test)), norm(self.embed_all(ctx))
        qk_kwargs = {"qs_s": s_test, "ks_s": s_ctx}
        kk_kwargs = {"qs_s": s_ctx, "ks_s": s_ctx}
        if self.dist is not None:
            vdist = jit(vmap(self.dist))
            d_qk, d_kk = vdist(s_test, s_ctx), vdist(s_ctx, s_ctx)
            # NOTE: this assumes the sentinal for a masked value is an
            # infinite value in any entry of the last dim of `d`
            to_mask = jit(lambda d: jnp.any(jnp.isfinite(d), axis=-1))
            d_qk_mask, d_kk_mask = to_mask(d_qk), to_mask(d_kk)
        for _ in range(self.num_blks):
            blk = self.blk.copy()  # new bias for every blk & rep
            for _ in range(self.num_reps):
                if self.bias is not None:
                    bias = self.bias.copy()
                    qk_kwargs["bias"] = bias(d_qk, d_qk_mask)
                    kk_kwargs["bias"] = bias(d_kk, d_kk_mask)
                qvs, kvs = blk(qvs, kvs, mask_ctx, training, qk_kwargs, kk_kwargs)
        qvs = self.norm.copy()(qvs)
        output = self.head(qvs, training)
        return self.output_fn(output)


class ScanTNPKR(nn.Module):
    """Transformer Neural Process - Kernel Regression (TNP-KR) using memory efficient
    scanning.

    Args:
        num_blks: Number of `KRBlocks` to use.
        num_reps: Number of times to repeat each `KRBlock`.
        embed_s: A module that embeds the index set prior to embedding with
            function values.
        embed_f: A module that embeds function values prior to embedding with
            the index set.
        embed_obs: A module that creates embeddings for observed (context) and
            unobserved (test) points.
        embed_all: A module that jointly embeds `obs`, `s`, and `f` embeddings.
        blk: A block to use for each layer and each repetition.
        norm: A normalization module used in `KRBlocks`.
        head: Transforms the tokens into model output.
        output_fn: A function that transforms the model output into
            a form that can be consumed by loss functions.
        train_step: What training step to use.
        valid_step: What validation step to use.

    Returns:
        An instance of the `TNP-KR` model.
    """

    num_blks: int = 6
    num_reps: int = 1
    embed_s: Callable = lambda x: x
    embed_f: Callable = lambda x: x
    embed_obs: nn.Module = nn.Embed(2, 4)
    embed_all: nn.Module = MLP([256, 128, 64], nn.gelu)
    blk: nn.Module = KRBlock(MultiHeadAttention(RBFNetworkBiasedScanAttention()))
    norm: nn.Module = nn.LayerNorm()
    head: nn.Module = MLP([256, 64, 2], nn.gelu)
    output_fn: Callable = DiagonalMVNOutput.from_conditional_np
    train_step: Callable = likelihood_train_step
    valid_step: Callable = likelihood_valid_step

    @nn.compact
    def __call__(
        self,
        s_ctx: jax.Array,  # [B, L_ctx, D_s]
        f_ctx: jax.Array,  # [B, L_ctx, D_f]
        s_test: jax.Array,  # [B, L_test, D_s]
        mask_ctx: Optional[jax.Array] = None,  # [B, L_ctx]
        training: bool = False,
        **kwargs,
    ):
        r"""Run module forward.

        Args:
            rng: A psuedo-random number generator.
            s_ctx: An index set array of shape `[B, L_ctx, D_S]` where
                `B` is batch size, `L_ctx` is number of context
                locations, and `L_ctx` is the dimension of each location.
            f_ctx: A function value array of shape `[B, L_ctx, D_F]` where `B` is
                batch size, `L_ctx` is number of context locations, and `D_F` is
                the dimension of each function value.
            s_test: A location array of shape `[B, L_test, D_S]` where `B` is
                batch size, `L_test` is number of test locations, and `D_S`
                is the dimension of each location.
            mask_ctx: An optional array of shape `[B, L_ctx]`
                valid positions for each `L_ctx` sequence in the batch.
            training: A boolean indicating whether this call is performed during
                training.

        Returns:
            $\mu_f,\sigma_f\in\mathbb{R}^{B\times L_\text{test}\times D_F}$.
        """
        stack = lambda *args: jnp.concatenate([x for x in args if x.size > 0], axis=-1)
        f_test = jnp.zeros([*s_test.shape[:-1], f_ctx.shape[-1]])
        obs = jnp.ones(f_ctx.shape[:-1], dtype=jnp.uint8)
        unobs = jnp.zeros(f_test.shape[:-1], dtype=jnp.uint8)
        ctx = stack(self.embed_obs(obs), self.embed_s(s_ctx), self.embed_f(f_ctx))
        test = stack(self.embed_obs(unobs), self.embed_s(s_test), self.embed_f(f_test))
        qvs, kvs = self.norm(self.embed_all(test)), self.norm(self.embed_all(ctx))
        qk_kwargs = {"qs_s": s_test, "ks_s": s_ctx}
        kk_kwargs = {"qs_s": s_ctx, "ks_s": s_ctx}
        for _ in range(self.num_blks):
            blk = self.blk.copy()
            for _ in range(self.num_reps):
                qvs, kvs = blk(qvs, kvs, mask_ctx, training, qk_kwargs, kk_kwargs)
        qvs = self.norm.copy()(qvs)
        output = self.head(qvs, training)
        return self.output_fn(output)
