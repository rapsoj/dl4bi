import jax.numpy as jnp
from jax import random

from dl4bi.core.utils import bootstrap2, mask_from_valid_lens


def test_bootstrap():
    B, L, D, K = 4, 37, 8, 4
    rng = random.key(42)
    valid_lens = jnp.array([12, 17, 37, 26])
    mask = mask_from_valid_lens(L, valid_lens)
    x = random.normal(rng, (B, L, D))
    x_boot, mask_boot = bootstrap2(rng, x, mask, K)
    assert x_boot.shape == (B, K, L, D)
    assert mask_boot.shape == (B, K, L)
    assert jnp.isfinite(x_boot.mean(where=mask_boot[..., None]))
