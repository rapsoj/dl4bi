from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
from jax import jit, random, vmap


@partial(jit, static_argnames=("independent",))
def permute_L_in_BLD(
    rng: jax.Array,
    independent: bool = False,
    *arrays,
):
    B, L = arrays[0].shape[:2]
    if independent:
        rngs = random.split(rng, B)
        permute_idx = _vpermute_idx(rngs, L)
        inv_permute_idx = jnp.argsort(permute_idx, axis=1)
        vpermute = vmap(lambda v, idx: v[:, idx])
        arrays = [vpermute(a, permute_idx) for a in arrays]
        return *arrays, inv_permute_idx
    permute_idx = _permute_idx(rng, L)
    inv_permute_idx = jnp.argsort(permute_idx)
    arrays = [a[:, permute_idx] for a in arrays]
    return *arrays, inv_permute_idx


def _vpermute_idx(rngs: jax.Array, L: int):
    return vmap(lambda rng: _permute_idx(rng, L))(rngs)


def _permute_idx(rng: jax.Array, L: int):
    return random.choice(rng, L, (L,), replace=False)


@jit
def inv_permute_L_in_BLD(inv_permute_idx: jax.Array, *arrays):
    if inv_permute_idx.ndim == 1:
        return [a[:, inv_permute_idx] for a in arrays]
    v_inv = vmap(lambda v, idx: v[:, idx])
    return [v_inv(a, inv_permute_idx) for a in arrays]


@partial(
    jit,
    static_argnames=(
        "num_ctx_min",
        "num_ctx_max",
        "num_test",
        "test_includes_ctx",
    ),
)
def batch_BLD(
    rng: jax.Array,
    inv_permute_idx: jax.Array,
    num_ctx_min: int,
    num_ctx_max: int,
    num_test: int,
    test_includes_ctx: bool = False,
    *arrays,
):
    B = arrays[0].shape[0]
    valid_lens_ctx = random.randint(rng, (B,), num_ctx_min, num_ctx_max)
    valid_lens_test = jnp.repeat(num_test, B)
    ctx = [a[:, :num_ctx_max] for a in arrays]
    if test_includes_ctx:
        test = [a[:, :num_test] for a in arrays]
    else:
        test = [a[:, num_ctx_max : num_ctx_max + num_test] for a in arrays]
    return (
        *ctx,
        valid_lens_ctx,
        *test,
        valid_lens_test,
        inv_permute_idx,
        test_includes_ctx,
    )


@partial(jit, static_argnames=("test_includes_ctx",))
def unbatch_BLD(
    ctx: Sequence[jax.Array],
    valid_lens_ctx: jax.Array,
    test: Sequence[jax.Array],
    valid_lens_test: jax.Array,
    inv_permute_idx: jax.Array,
    test_includes_ctx: bool,
):
    L = inv_permute_idx.shape[0]
    if inv_permute_idx.ndim > 1:
        L = inv_permute_idx.shape[1]
    if test_includes_ctx:
        arrays = [_nan_pad(a, axis=1, L=L) for a in test]
        return *arrays, inv_permute_idx
    arrays = [jnp.concat(pair, axis=1) for pair in zip(ctx, test)]
    arrays = [_nan_pad(a, axis=1, L=L) for a in arrays]
    return *arrays, inv_permute_idx


def _nan_pad(v: jax.Array, axis: int, L: int):
    pad = [(0, 0)] * v.ndim
    L_v = v.shape[axis]
    pad[axis] = (0, L - L_v)
    return jnp.pad(v, pad, mode="constant", constant_values=jnp.nan)
