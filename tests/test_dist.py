import jax.numpy as jnp
from jax import random, vmap

from dl4bi.core import k_nearest_senders, mask_from_valid_lens, scipy_k_nearest_senders


def test_k_nearest_senders():
    rng = random.key(55)
    B, L, S, K = 4, 128, 2, 16
    r = random.normal(rng, (B, L, S))
    v = random.randint(rng, (B,), 0, L)
    m = mask_from_valid_lens(L, v)
    s = jnp.where(m, r, 1e6)
    idx, d = vmap(lambda r, s: k_nearest_senders(r, s, K))(r, s)
    idx_s, d_s = vmap(lambda r, s: scipy_k_nearest_senders(r, s, K))(r, s)
    assert (idx == idx_s).all(), "Indices do not match!"
    assert jnp.allclose(d, d_s), "Distances are not close!"
