"""
Provides standard containers for Tabular Data.

This is based on the following general state machine:

1. Data.permute() -> PermutedData
2. PermutedData.batch() -> DenseBatchedData [data loss: non-context, non-test points dropped]
3. DenseBatchedData.sparse() -> SparseBatchedData [no-op here since data is already packed]

4. SparseBatchedData.dense() -> DenseBatchedData [no-op here since sparse = dense data]
5. DenseBatchedData.unbatch() -> PermutedData [dropped points filled with NaNs]
6. PermutedData.inv_permute() -> Data
"""

from dataclasses import dataclass

import jax

from .utils import (
    MetaLearningBatch,
    MetaLearningData,
    batch_BLD,
    inv_permute_L_in_BLD,
    permute_L_in_BLD,
    unbatch_BLD,
)


@dataclass(frozen=True)
class TabularData(MetaLearningData):
    """A simple `TabularData` container."""

    x: jax.Array  # [B, L, D_x]
    f: jax.Array  # [B, L, D_f]

    def permute(self, rng: jax.Array, independent: bool = False):
        """Returns a `PermutedTabularData` object.

        Args:
            rng: A PRNG for generating the permutation indices.
            independent: Whether to permute each element in the
                batch separately.
        """
        args = permute_L_in_BLD(rng, [self.x, self.f], independent)
        return PermutedTabularData(*args)


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    TabularData,
    lambda d: ((d.x, d.f), None),
    lambda _aux, children: TabularData(*children),
)


@dataclass(frozen=True)
class PermutedTabularData(MetaLearningData):
    """A permuted version of `TabularData`."""

    x: jax.Array  # [B, L, D_x]
    f: jax.Array  # [B, L, D_f]
    inv_permute_idx: jax.Array  # [B, L] or [L]

    def inv_permute(self):
        """Inverts the permutation and returns `TabularData`."""
        x, f = inv_permute_L_in_BLD([self.x, self.f], self.inv_permute_idx)
        return TabularData(x, f)

    def batch(
        self,
        rng: jax.Array,
        num_ctx_min: int,
        num_ctx_max: int,
        num_test: int,
        test_includes_ctx: bool = False,
    ):
        args = batch_BLD(
            rng,
            [self.x, self.f],
            num_ctx_min,
            num_ctx_max,
            num_test,
            test_includes_ctx,
        )
        kwargs = {
            "test_includes_ctx": test_includes_ctx,
            "inv_permute_idx": self.inv_permute_idx,
        }
        return DenseBatchedTabularData(*args, **kwargs)


jax.tree_util.register_pytree_node(
    PermutedTabularData,
    lambda d: ((d.x, d.f, d.inv_permute_idx), None),
    lambda _aux, children: PermutedTabularData(*children),
)


@dataclass(frozen=True)
class DenseBatchedTabularData(MetaLearningBatch):
    x_ctx: jax.Array
    f_ctx: jax.Array
    valid_lens_ctx: jax.Array
    x_test: jax.Array
    f_test: jax.Array
    valid_lens_test: jax.Array
    inv_permute_idx: jax.Array
    test_includes_ctx: bool

    def unbatch(self):
        i = self.inv_permute_idx
        L = i.shape[0] if i.ndim == 1 else i.shape[1]
        x, f = unbatch_BLD(
            [self.x_ctx, self.f_ctx],
            [self.x_test, self.f_test],
            L,
            self.test_includes_ctx,
        )
        return PermutedTabularData(x, f, self.inv_permute_idx)

    def sparse(self):
        return SparseBatchedTabularData(
            self.x_ctx,
            self.f_ctx,
            self.valid_lens_ctx,
            self.x_test,
            self.f_test,
            self.valid_lens_test,
            self.inv_permute_idx,
            self.test_includes_ctx,
        )


jax.tree_util.register_pytree_node(
    DenseBatchedTabularData,
    lambda d: (
        (
            d.x_ctx,
            d.f_ctx,
            d.valid_lens_ctx,
            d.x_test,
            d.f_test,
            d.valid_lens_test,
            d.inv_permute_idx,
        ),
        (d.test_includes_ctx,),
    ),
    lambda aux, children: DenseBatchedTabularData(*children, *aux),
)


@dataclass(frozen=True)
class SparseBatchedTabularData(MetaLearningBatch):
    x_ctx: jax.Array
    f_ctx: jax.Array
    valid_lens_ctx: jax.Array
    x_test: jax.Array
    f_test: jax.Array
    valid_lens_test: jax.Array
    inv_permute_idx: jax.Array
    test_includes_ctx: bool

    def dense(self):
        return DenseBatchedTabularData(
            self.x_ctx,
            self.f_ctx,
            self.valid_lens_ctx,
            self.x_test,
            self.f_test,
            self.valid_lens_test,
            self.inv_permute_idx,
            self.test_includes_ctx,
        )


jax.tree_util.register_pytree_node(
    SparseBatchedTabularData,
    lambda d: (
        (
            d.x_ctx,
            d.f_ctx,
            d.valid_lens_ctx,
            d.x_test,
            d.f_test,
            d.valid_lens_test,
            d.inv_permute_idx,
        ),
        (d.test_includes_ctx,),
    ),
    lambda aux, children: SparseBatchedTabularData(*children, *aux),
)
