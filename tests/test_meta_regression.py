import jax
import jax.numpy as jnp
from jax import random

from dsp.meta_regression import (
    ANP,
    BANP,
    BNP,
    CANP,
    CNP,
    DKR,
    NP,
    TNPD,
    TNPDS,
    TNPND,
    ConvCNP,
    SPTx,
)


def test_meta_regression():
    B, L = 4, 10
    key = random.key(42)
    rng_data, rng_params, rng_dropout, rng_extra = random.split(key, 4)
    s = jnp.linspace(0, 1.0, L)
    s = jnp.repeat(s[None, :, None], B, axis=0)  # [B, S, D_s=1]
    valid_lens = jnp.array([2, 4, 9, 3])
    f = random.normal(rng_data, s.shape)
    for np in [ANP, BNP, BANP, CANP, CNP, DKR, NP, TNPD, TNPDS, TNPND, ConvCNP, SPTx]:
        m = np()
        (f_mu, f_std, *_), params = m.init_with_output(
            {"params": rng_params, "dropout": rng_dropout, "extra": rng_extra},
            s_ctx=s,
            f_ctx=f,
            s_test=s,
            valid_lens_ctx=valid_lens,
            valid_lens_test=valid_lens,
            training=True,
        )
        K = f_mu.shape[0] // f.shape[0]
        assert f_mu.shape == (B * K, L, 1)


def test_meta_regression_data_leaks():
    B, L, N = 4, 10, 5
    key = random.key(42)
    rng_data, rng_params, rng_dropout, rng_extra = random.split(key, 4)
    s = jnp.linspace(0, 1.0, L)
    s = jnp.repeat(s[None, :, None], B, axis=0)  # [B, S, D_s=1]
    valid_lens_ctx = jnp.array([N] * B)
    valid_lens_test = jnp.array([L] * B)
    f = random.normal(rng_data, s.shape)
    # set second half to 0s (different from using half the array because of attn)
    s2 = s.at[:, N:, :].set(jnp.zeros((B, L - N, 1)))
    f2 = f.at[:, N:, :].set(jnp.zeros((B, L - N, 1)))
    for np in [ANP, BNP, BANP, CANP, CNP, DKR, NP, TNPD, TNPDS, TNPND, ConvCNP, SPTx]:
        print(np)
        m = np()
        (f_mu, f_std, *_), params = m.init_with_output(
            {"params": rng_params, "dropout": rng_dropout, "extra": rng_extra},
            s_ctx=s,
            f_ctx=f,
            s_test=s,
            valid_lens_ctx=valid_lens_ctx,
            valid_lens_test=valid_lens_test,
        )
        f_mu_half, f_std_half, *_ = m.apply(
            params,
            s_ctx=s2,
            f_ctx=f2,
            s_test=s,
            valid_lens_ctx=valid_lens_ctx,
            valid_lens_test=valid_lens_test,
            rngs={"dropout": rng_dropout, "extra": rng_extra},
        )
        assert jnp.allclose(f_mu, f_mu_half)
        assert jnp.allclose(f_std, f_std_half)
