from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, random, vmap
from jax.tree_util import Partial


def mask_from_valid_lens(max_len: int, valid_lens: jax.Array):
    """Return a boolean mask using `valid_lens`."""
    return jnp.arange(max_len) < valid_lens[..., None]


def pad_concat(x: jax.Array, y: jax.Array):
    """Concat channels of two layers, padding smaller spatial layer with zeros.

    Args:
        x: Tensor of shape [B, L_x, C_x].
        y: Tensor of shape [B, L_y, C_y].

    Returns:
        A concatenated tensor of shape [B, max(L_x, L_y), C_x + C_y].
    """
    L_x, L_y = x.shape[1], y.shape[1]
    padding = np.abs(L_x - L_y)
    is_even = padding % 2 == 0
    half_even = (padding // 2, padding // 2)
    half_odd = ((padding - 1) // 2, (padding + 1) // 2)
    p_e = Partial(jnp.pad, pad_width=((0, 0), half_even, (0, 0)), mode="reflect")
    p_o = Partial(jnp.pad, pad_width=((0, 0), half_odd, (0, 0)), mode="reflect")
    if L_x > L_y:
        y = p_e(y) if is_even else p_o(y)
    elif L_y > L_x:
        x = p_e(x) if is_even else p_o(x)
    return jnp.concatenate([x, y], axis=-1)


@partial(jit, static_argnames=("num_samples"))
def bootstrap_from_valid_lens(
    rng: jax.Array,
    x: jax.Array,  # [B, L, D]
    valid_lens: jax.Array,  # [B]
    num_samples: int = 1,
):
    """Bootstrap selects the first `valid_lens` values of `x` `num_samples` times.

    Args:
        rng: A PRNGKey.
        x: Array to bootstrap.
        valid_lens: The valid entries for every sequence in x.

    Returns:
        A bootstrap sampled array of shape [B * num_samples, L, D].
    """
    (B, L, _), K = x.shape, num_samples
    x = jnp.repeat(x, K, axis=0)
    valid_lens = jnp.repeat(valid_lens, K, axis=0)
    mask = mask_from_valid_lens(L, valid_lens).squeeze()
    rnd_idx = random.randint(rng, (B * K, L), 0, valid_lens[:, None])
    ord_idx = jnp.repeat(jnp.arange(L)[None, :], B * K, axis=0)
    boot_idx = mask * rnd_idx + ~mask * ord_idx
    return vmap(lambda row, idx: row[idx], (0, 0))(x, boot_idx), valid_lens


def bootstrap(
    rng: jax.Array,
    x: jax.Array,  # [B, L, D]
    mask: Optional[jax.Array],  # [B, L]
    num_samples: int,
):
    B = x.shape[0]
    rngs = random.split(rng, B)
    xs, masks = [], []
    for i in range(B):  # can't vmap variable concrete arrays
        mask_i = None if mask is None else mask[i]
        x_b, mask_b = _bootstrap_one(rngs[i], x[i], mask_i, num_samples)
        xs += [x_b]
        masks += [mask_b]
    return jnp.array(xs), jnp.array(masks)  # [B, K, L, D]


def _bootstrap_one(
    rng: jax.Array,
    x_i: jax.Array,  # [L, D]
    mask_i: Optional[jax.Array],  # [L] or None
    num_samples: int,
):
    idx = jnp.arange(x_i.shape[0]) if mask_i is None else jnp.where(mask_i)[0]
    K, L, L_valid = num_samples, x_i.shape[0], idx.shape[0]
    idx = random.choice(rng, idx, shape=(K, L_valid))
    x_boot = x_i[idx]  # x_k: [K, L_valid, D]
    x_boot = nan_pad(x_boot, axis=1, L=L)  # [K, L, D]
    mask_boot = mask_from_valid_lens(L, jnp.repeat(L_valid, K))
    return x_boot, mask_boot  # [K, L, D]


def nan_pad(v: jax.Array, axis: int, L: int):
    pad = [(0, 0)] * v.ndim
    L_v = v.shape[axis]
    pad[axis] = (0, L - L_v)
    return jnp.pad(v, pad, mode="constant", constant_values=jnp.nan)


def breakpoint_if_nonfinite(x):
    """Create a breakpoint when non-finite values in `x`."""
    is_finite = jnp.isfinite(x).all()

    def true_fn(x):
        pass

    def false_fn(x):
        jax.debug.breakpoint()

    lax.cond(is_finite, true_fn, false_fn, x)
