"""
Provides standard containers for Spatial Data.

This is based on the following general state machine:

1. Data.permute() -> PermutedData
2. PermutedData.batch() -> DenseBatchedData [data loss: non-context, non-test points dropped]
3. DenseBatchedData.sparse() -> SparseBatchedData [no-op here since data is already packed]

4. SparseBatchedData.dense() -> DenseBatchedData [no-op here since sparse = dense data]
5. DenseBatchedData.unbatch() -> PermutedData [dropped points filled with NaNs]
6. PermutedData.inv_permute() -> Data
"""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp

from .utils import (
    MetaLearningBatch,
    MetaLearningData,
    batch_BLD,
    flatten_spatial,
    inv_permute_L_in_BLD,
    permute_L_in_BLD,
    unbatch_BLD,
)


@dataclass(frozen=True, eq=False)
class SpatialData(MetaLearningData):
    """A simple `SpatialData` container."""

    x: Optional[jax.Array]  # [B, [S]+, D_x] or [B, D_x] or None
    s: jax.Array  # [B, [S]+, D_s]
    f: jax.Array  # [B, [S]+, D_f]

    def permute(self, rng: jax.Array, independent: bool = False):
        """Returns a `PermutedSpatialData` object.

        Args:
            rng: A PRNG for generating the permutation indices.
            independent: Whether to permute each element in the
                batch separately.
        """
        _s = flatten_spatial(self.s)
        _f = flatten_spatial(self.f)
        x = self.x
        # no need to inv_permute x if it is None or is the same value for all locations
        if self.x is not None and self.x.ndim != 2:
            _x = flatten_spatial(self.x)
            x, s, f, inv_permute_idx = permute_L_in_BLD(rng, [_x, _s, _f], independent)
            x = x.reshape(self.x.shape)
        else:
            s, f, inv_permute_idx = permute_L_in_BLD(rng, [_s, _f], independent)
        s = s.reshape(self.s.shape)
        f = f.reshape(self.f.shape)
        return PermutedSpatialData(x, s, f, inv_permute_idx)


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    SpatialData,
    lambda d: ((d.x, d.s, d.f), None),
    lambda _aux, children: SpatialData(*children),
)


@dataclass(frozen=True, eq=False)
class PermutedSpatialData(MetaLearningData):
    """A permuted version of `SpatialData`."""

    x: Optional[jax.Array]  # [B, [S]+, D_x] or [B, D_x] or None
    s: jax.Array  # [B, [S]+, D_s]
    f: jax.Array  # [B, [S]+, D_f]
    inv_permute_idx: jax.Array  # [B, L] or [L]

    def inv_permute(self):
        """Inverts the permutation and returns `SpatialData`."""
        _s = flatten_spatial(self.s)
        _f = flatten_spatial(self.f)
        x = self.x
        # no need to inv_permute x if it is None or is the same value for all locations
        if self.x is not None and self.x.ndim != 2:
            _x = flatten_spatial(self.x)
            x, s, f = inv_permute_L_in_BLD([_x, _s, _f], self.inv_permute_idx)
            x = x.reshape(self.x.shape)
        else:
            s, f = inv_permute_L_in_BLD([_s, _f], self.inv_permute_idx)
        s = s.reshape(self.s.shape)
        f = f.reshape(self.f.shape)
        return SpatialData(x, s, f)

    def batch(
        self,
        rng: jax.Array,
        num_ctx_min: int,
        num_ctx_max: int,
        num_test: int,
        test_includes_ctx: bool = False,
    ):
        batch_BLD_args = (num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
        kwargs = {
            "shape_s": self.s.shape,
            "broadcast_x": False,
            "inv_permute_idx": self.inv_permute_idx,
            "test_includes_ctx": test_includes_ctx,
        }
        s = flatten_spatial(self.s)
        f = flatten_spatial(self.f)
        if self.x is None:
            args = batch_BLD(rng, [s, f], *batch_BLD_args)
            # pack x=None back into args
            s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, *rest = args
            args = (None, s_ctx, f_ctx, valid_lens_ctx, None, s_test, f_test, *rest)
            return DenseBatchedSpatialData(*args, **kwargs)
        if self.x.ndim == 2:
            x = jnp.broadcast_to(self.x[:, None], (*s.shape[:-1], self.x.shape[-1]))
            kwargs["broadcast_x"] = True
        else:
            x = flatten_spatial(self.x)
        args = batch_BLD(rng, [x, s, f], *batch_BLD_args)
        return DenseBatchedSpatialData(*args, **kwargs)


jax.tree_util.register_pytree_node(
    PermutedSpatialData,
    lambda d: ((d.x, d.s, d.f, d.inv_permute_idx), None),
    lambda _aux, children: PermutedSpatialData(*children),
)


@dataclass(frozen=True, eq=False)
class DenseBatchedSpatialData(MetaLearningBatch):
    x_ctx: Optional[jax.Array]
    s_ctx: jax.Array
    f_ctx: jax.Array
    valid_lens_ctx: jax.Array
    x_test: Optional[jax.Array]
    s_test: jax.Array
    f_test: jax.Array
    valid_lens_test: jax.Array
    inv_permute_idx: jax.Array
    test_includes_ctx: bool
    shape_s: tuple
    broadcast_x: bool

    def unbatch(self):
        i = self.inv_permute_idx
        L = i.shape[0] if i.ndim == 1 else i.shape[1]
        unbatch_args = (L, self.test_includes_ctx)
        if self.x_ctx is None or self.broadcast_x:
            s, f = unbatch_BLD(
                [self.s_ctx, self.f_ctx],
                [self.s_test, self.f_test],
                *unbatch_args,
            )
            s = s.reshape(self.shape_s)
            f = f.reshape(*self.shape_s[:-1], self.f_ctx.shape[-1])
            x = None
            if self.broadcast_x:
                # since x is the same for all L, select first x for each batch element
                x = self.x_ctx[:, 0]
            return PermutedSpatialData(x, s, f, self.inv_permute_idx)
        x, s, f = unbatch_BLD(
            [self.x_ctx, self.s_ctx, self.f_ctx],
            [self.x_test, self.s_test, self.f_test],
            *unbatch_args,
        )
        x = x.reshape(*self.shape_s[:-1], self.x_ctx.shape[-1])
        s = s.reshape(self.shape_s)
        f = f.reshape(*self.shape_s[:-1], self.f_ctx.shape[-1])
        return PermutedSpatialData(x, s, f, self.inv_permute_idx)

    def sparse(self):
        return SparseBatchedSpatialData(
            self.x_ctx,
            self.s_ctx,
            self.f_ctx,
            self.valid_lens_ctx,
            self.x_test,
            self.s_test,
            self.f_test,
            self.valid_lens_test,
            self.inv_permute_idx,
            self.test_includes_ctx,
            self.shape_s,
            self.broadcast_x,
        )


jax.tree_util.register_pytree_node(
    DenseBatchedSpatialData,
    lambda d: (
        (
            d.x_ctx,
            d.s_ctx,
            d.f_ctx,
            d.valid_lens_ctx,
            d.x_test,
            d.s_test,
            d.f_test,
            d.valid_lens_test,
            d.inv_permute_idx,
        ),
        (d.test_includes_ctx, d.shape_s, d.broadcast_x),
    ),
    lambda aux, children: DenseBatchedSpatialData(*children, *aux),
)


@dataclass(frozen=True, eq=False)
class SparseBatchedSpatialData(MetaLearningBatch):
    x_ctx: Optional[jax.Array]
    s_ctx: jax.Array
    f_ctx: jax.Array
    valid_lens_ctx: jax.Array
    x_test: Optional[jax.Array]
    s_test: jax.Array
    f_test: jax.Array
    valid_lens_test: jax.Array
    inv_permute_idx: jax.Array
    test_includes_ctx: bool
    shape_s: tuple
    broadcast_x: bool

    def dense(self):
        return DenseBatchedSpatialData(
            self.x_ctx,
            self.s_ctx,
            self.f_ctx,
            self.valid_lens_ctx,
            self.x_test,
            self.s_test,
            self.f_test,
            self.valid_lens_test,
            self.inv_permute_idx,
            self.test_includes_ctx,
            self.shape_s,
            self.broadcast_x,
        )


jax.tree_util.register_pytree_node(
    SparseBatchedSpatialData,
    lambda d: (
        (
            d.x_ctx,
            d.s_ctx,
            d.f_ctx,
            d.valid_lens_ctx,
            d.x_test,
            d.s_test,
            d.f_test,
            d.valid_lens_test,
            d.inv_permute_idx,
        ),
        (d.test_includes_ctx, d.shape_s, d.broadcast_x),
    ),
    lambda aux, children: SparseBatchedSpatialData(*children, *aux),
)
