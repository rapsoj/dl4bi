from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import jit, random

from .utils import (
    MetaLearningBatch,
    MetaLearningData,
    batch_BLD,
    permute_L_in_BLD,
)


@dataclass(frozen=True, eq=False)
class TemporalData(MetaLearningData):
    """A simple `TabularData` container."""

    x: Optional[jax.Array]  # [B, T, D_x] or [B, D_x] or None
    t: jax.Array  # [B, T]
    f: jax.Array  # [B, T, D_f]

    def batch(
        self,
        rng: jax.Array,
        num_ctx_min: int,
        num_ctx_max: int,
        num_test: int,
        test_includes_ctx: bool = False,
    ):
        return _batch(
            rng,
            self.x,
            self.t,
            self.f,
            num_ctx_min,
            num_ctx_max,
            num_test,
            test_includes_ctx,
        )


@partial(
    jit,
    static_argnames=(
        "num_ctx_min",
        "num_ctx_max",
        "num_test",
        "test_includes_ctx",
    ),
)
def _batch(
    rng,
    x: Optional[jax.Array],
    t: jax.Array,
    f: jax.Array,
    num_ctx_min: int,
    num_ctx_max: int,
    num_test: int,
    test_includes_ctx: bool,
):
    rng_p, rng_b = random.split(rng)
    has_x = x is not None
    broadcast_x = x.ndim == 2 if has_x else None
    batch_args = (num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    t = t[..., None]
    if broadcast_x or not has_x:
        x_ctx = x_test = x
        t, f, inv_permute_idx = permute_L_in_BLD(rng_p, [t, f])
        args = batch_BLD(rng_b, [t, f], *batch_args)
        t_ctx, f_ctx, v_ctx, t_test, f_test, *rest = args
        if broadcast_x:
            x_ctx = jnp.repeat(x_ctx[:, None], num_ctx_max, axis=1)
            x_test = jnp.repeat(x_test[:, None], num_test, axis=1)
        args = (x_ctx, t_ctx, f_ctx, v_ctx, x_test, t_test, f_test, *rest)
    else:
        x, t, f, inv_permute_idx = permute_L_in_BLD(rng_p, [x, t, f])
        args = batch_BLD(rng_b, [x, t, f], *batch_args)
    return TemporalBatch(*args, inv_permute_idx=inv_permute_idx)


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    TemporalData,
    lambda d: ((d.x, d.t, d.f), None),
    lambda _aux, children: TemporalData(*children),
)


@dataclass(frozen=True, eq=False)
class TemporalBatch(MetaLearningBatch):
    x_ctx: Optional[jax.Array]  # [B, L_ctx, D_x] or None
    t_ctx: jax.Array  # [B, L_ctx, 1]
    f_ctx: jax.Array  # [B, L_ctx, D_f]
    valid_lens_ctx: jax.Array  # [B]
    x_test: Optional[jax.Array]  # [B, L_test, D_x]
    t_test: jax.Array  # [B, L_test, 1]
    f_test: jax.Array  # [B, L_test, D_f]
    valid_lens_test: jax.Array  # [B]
    inv_permute_idx: jax.Array  # [L]

    def plot_1d(self, model_output):
        # TODO(danj): IMPLEMENT
        pass


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    TemporalBatch,
    lambda d: (
        (
            d.x_ctx,
            d.t_ctx,
            d.f_ctx,
            d.valid_lens_ctx,
            d.x_test,
            d.t_test,
            d.f_test,
            d.valid_lens_test,
            d.inv_permute_idx,
        ),
        None,
    ),
    lambda _aux, children: TemporalBatch(*children),
)
