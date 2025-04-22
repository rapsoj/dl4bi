from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
from jax import jit, random, vmap

from .utils import MetaLearningBatch, MetaLearningData, batch_BLD


@dataclass(frozen=True, eq=False)
class TabularData(MetaLearningData):
    """A simple `TabularData` container."""

    x: jax.Array  # [N, D_x]
    f: jax.Array  # [N, D_f]

    def batch(
        self,
        rng: jax.Array,
        num_ctx_min: int,
        num_ctx_max: int,
        num_test: int,
        obs_noise: Optional[float] = None,
        test_includes_ctx: bool = False,
        batch_size: int = 64,
    ):
        return _batch(
            rng,
            self.x,
            self.f,
            num_ctx_min,
            num_ctx_max,
            num_test,
            obs_noise,
            test_includes_ctx,
            batch_size,
        )


@partial(
    jit,
    static_argnames=(
        "num_ctx_min",
        "num_ctx_max",
        "num_test",
        "test_includes_ctx",
        "obs_noise",
        "batch_size",
    ),
)
def _batch(
    rng: jax.Array,
    x: jax.Array,  # [N, D_x]
    f: jax.Array,  # [N, D_f]
    num_ctx_min: int,
    num_ctx_max: int,
    num_test: int,
    obs_noise: Optional[float] = None,
    test_includes_ctx: bool = False,
    batch_size: int = 64,
):
    N, B, L = x.shape[0], batch_size, num_ctx_max + num_test
    rng_i, rng_eps, rng_b = random.split(rng, 3)
    rng_bs = random.split(rng_b, B)
    idx = vmap(lambda rng: random.choice(rng, N, (L,), replace=False))(rng_bs)
    x_batch, f_batch = x[idx], f[idx]  # [B, L, D_{x,f}]
    x_ctx, f_ctx, mask_ctx, x_test, f_test, mask_test = batch_BLD(
        rng_b,
        [x_batch, f_batch],
        num_ctx_min,
        num_ctx_max,
        num_test,
        test_includes_ctx,
    )
    if obs_noise:
        f_ctx += obs_noise * random.normal(rng_eps, f_ctx.shape)
    return TabularBatch(x_ctx, f_ctx, mask_ctx, x_test, f_test, mask_test)


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    TabularData,
    lambda d: ((d.x, d.f), None),
    lambda _aux, children: TabularData(*children),
)


@dataclass(frozen=True, eq=False)
class TabularBatch(MetaLearningBatch):
    x_ctx: jax.Array
    f_ctx: jax.Array
    mask_ctx: Optional[jax.Array]
    x_test: jax.Array
    f_test: jax.Array
    mask_test: Optional[jax.Array] = None


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    TabularBatch,
    lambda d: (
        (
            d.x_ctx,
            d.f_ctx,
            d.mask_ctx,
            d.x_test,
            d.f_test,
            d.mask_test,
        ),
        None,
    ),
    lambda _aux, children: TabularBatch(*children),
)
