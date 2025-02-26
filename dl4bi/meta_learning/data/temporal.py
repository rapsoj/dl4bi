"""
Provides standard containers for Temporal Data.

This is based on the following general state machine:

TemporalData.permute(...) -> PermutedTemporalData
PermutedTemporalData.inv_permute() -> TemporalData

PermutedTemporalData.batch(...) -> BatchedPermutedTemporalData [data loss: non-context, non-test points dropped]
BatchedPermutedTemporalData.unbatch() -> PermutedTemporalData [dropped points filled with NaNs]
BatchedPermutedTemporalData.inv_permute() -> BatchedTemporalData [dropped points filled with NaNs]
"""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp

from .utils import (
    MetaLearningBatch,
    MetaLearningData,
    batch_BLD,
    inv_permute_L_in_BLD,
    permute_L_in_BLD,
    unbatch_BLD,
)


@dataclass(frozen=True, eq=False)
class TemporalData(MetaLearningData):
    """A simple `TemporalData` container."""

    x: Optional[jax.Array]  # [B, T, D_x] or [B, D_x] or None
    t: jax.Array  # [B, T]
    f: jax.Array  # [B, T, D_f]

    def permute(self, rng: jax.Array, independent: bool = False):
        """Returns a `PermutedTemporalData` object.

        Args:
            rng: A PRNG for generating the permutation indices.
            independent: Whether to permute each element in the
                batch separately.
        """
        x = self.x
        # no need to inv_permute x if it is None or is the same value for all locations
        if self.x is not None and self.x.ndim != 2:
            x, s, f, inv_permute_idx = permute_L_in_BLD(
                rng, [self.x, self.t, self.f], independent
            )
            x = x.reshape(self.x.shape)
        else:
            s, f, inv_permute_idx = permute_L_in_BLD(rng, [self.t, self.f], independent)
        return PermutedTemporalData(x, s, f, inv_permute_idx)


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    TemporalData,
    lambda d: ((d.x, d.t, d.f), None),
    lambda _aux, children: TemporalData(*children),
)


@dataclass(frozen=True, eq=False)
class PermutedTemporalData(MetaLearningData):
    """A permuted version of `TemporalData`."""

    x: Optional[jax.Array]  # [B, T, D_x] or [B, D_x] or None
    t: jax.Array  # [B, T]
    f: jax.Array  # [B, T, D_f]
    inv_permute_idx: jax.Array  # [B, L] or [L]

    def inv_permute(self):
        """Inverts the permutation and returns `TemporalData`."""
        x = self.x
        # no need to inv_permute x if it is None or is the same value for all locations
        if self.x is not None and self.x.ndim != 2:
            x, s, f = inv_permute_L_in_BLD([x, self.t, self.f], self.inv_permute_idx)
        else:
            s, f = inv_permute_L_in_BLD([self.t, self.f], self.inv_permute_idx)
        return TemporalData(x, s, f)

    def batch(
        self,
        rng: jax.Array,
        num_ctx_min: int,
        num_ctx_max: int,
        num_test: int,
        test_includes_ctx: bool = False,
    ):
        x, t = self.x, self.t[..., None]
        batch_BLD_args = (num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
        kwargs = {
            "broadcast_x": False,
            "inv_permute_idx": self.inv_permute_idx,
            "test_includes_ctx": test_includes_ctx,
        }
        if self.x is None:
            args = batch_BLD(rng, [t, self.f], *batch_BLD_args)
            # pack x=None back into args
            s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, *rest = args
            args = (x, s_ctx, f_ctx, valid_lens_ctx, x, s_test, f_test, *rest)
            return BatchedPermutedTemporalData(*args, **kwargs)
        if self.x.ndim == 2:
            x = jnp.broadcast_to(self.x[:, None], (*self.t.shape, self.x.shape[-1]))
            kwargs["broadcast_x"] = True
        args = batch_BLD(rng, [x, t, self.f], *batch_BLD_args)
        return BatchedPermutedTemporalData(*args, **kwargs)


jax.tree_util.register_pytree_node(
    PermutedTemporalData,
    lambda d: ((d.x, d.t, d.f, d.inv_permute_idx), None),
    lambda _aux, children: PermutedTemporalData(*children),
)


@dataclass(frozen=True, eq=False)
class BatchedPermutedTemporalData(MetaLearningBatch):
    x_ctx: Optional[jax.Array]
    t_ctx: jax.Array
    f_ctx: jax.Array
    valid_lens_ctx: jax.Array
    x_test: Optional[jax.Array]
    t_test: jax.Array
    f_test: jax.Array
    valid_lens_test: jax.Array
    inv_permute_idx: jax.Array
    test_includes_ctx: bool
    broadcast_x: bool

    def unbatch(self):
        i = self.inv_permute_idx
        L = i.shape[0] if i.ndim == 1 else i.shape[1]
        unbatch_args = (L, self.test_includes_ctx)
        if self.x_ctx is None or self.broadcast_x:
            t, f = unbatch_BLD(
                [self.t_ctx, self.f_ctx],
                [self.t_test, self.f_test],
                *unbatch_args,
            )
            x = None
            if self.broadcast_x:
                # since x is the same for all L, select first x for each batch element
                x = self.x_ctx[:, 0]
            return PermutedTemporalData(x, t, f, self.inv_permute_idx)
        x, t, f = unbatch_BLD(
            [self.x_ctx, self.t_ctx, self.f_ctx],
            [self.x_test, self.t_test, self.f_test],
            *unbatch_args,
        )
        return PermutedTemporalData(x, t[..., 0], f, self.inv_permute_idx)


jax.tree_util.register_pytree_node(
    BatchedPermutedTemporalData,
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
        (d.test_includes_ctx, d.broadcast_x),
    ),
    lambda aux, children: BatchedPermutedTemporalData(*children, *aux),
)
