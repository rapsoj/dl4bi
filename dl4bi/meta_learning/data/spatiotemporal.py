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

from ...core.data import Batch, Data
from .utils import (
    batch_BLD,
    flatten_spatial,
    inv_permute_L_in_BLD,
    permute_L_in_BLD,
    unbatch_BLD,
)


@dataclass(frozen=True)
class SpatialData(Data):
    """A simple `SpatialData` container."""

    x: Optional[jax.Array]  # [B, [S]+, D_x] or [B, 1, D_x] or None
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
        # when x.shape[1]=1 means the same features are broadcast to all locations,
        # so no need to permute
        if self.x is not None and self.x.shape[1] != 1:
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


@dataclass(frozen=True)
class PermutedSpatialData(Data):
    """A permuted version of `SpatialData`."""

    x: Optional[jax.Array]  # [B, [S]+, D_x] or [B, 1, D_x] or None
    s: jax.Array  # [B, [S]+, D_s]
    f: jax.Array  # [B, [S]+, D_f]
    inv_permute_idx: jax.Array  # [B, L] or [L]

    def inv_permute(self):
        """Inverts the permutation and returns `SpatialData`."""
        _s = flatten_spatial(self.s)
        _f = flatten_spatial(self.f)
        x = self.x
        # no need to inv_permute x if it is None or is the same value for all locations
        if self.x is not None and self.x.shape[1] != 1:
            _x = flatten_spatial(self.x)
            x, s, f, inv_permute_idx = inv_permute_L_in_BLD(
                [_x, _s, _f], self.inv_permute_idx
            )
            x = x.reshape(self.x.shape)
        else:
            s, f, inv_permute_idx = inv_permute_L_in_BLD([_s, _f], self.inv_permute_idx)
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
        x = self.x
        batch_BLD_args = (
            self.inv_permute_idx,
            num_ctx_min,
            num_ctx_max,
            num_test,
            test_includes_ctx,
        )
        kwargs = {"shape_s": self.s.shape, "broadcast_x": False}
        if self.x is None:
            args = batch_BLD(rng, [self.s, self.f], *batch_BLD_args)
            # pack x=None back into args
            s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, *rest = args
            args = (x, s_ctx, f_ctx, valid_lens_ctx, x, s_test, f_test, *rest)
            return DenseBatchedSpatialData(*args, **kwargs)
        if self.x.shape[1] == 1:
            x = jnp.broadcast_to(self.x, (self.s.shape[:-1], self.x.shape[-1]))
            kwargs["broadcast_x"] = True
        args = batch_BLD(rng, [x, self.s, self.f], *batch_BLD_args)
        return DenseBatchedSpatialData(*args, **kwargs)


jax.tree_util.register_pytree_node(
    PermutedSpatialData,
    lambda d: ((d.x, d.s, d.f, d.inv_permute_idx), None),
    lambda _aux, children: PermutedSpatialData(*children),
)


@dataclass(frozen=True)
class DenseBatchedSpatialData(Batch):
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
        unbatch_args = (
            self.valid_lens_test,
            self.inv_permute_idx,
            self.test_includes_ctx,
        )
        if self.x_ctx is None:
            args = unbatch_BLD(
                [self.s_ctx, self.f_ctx],
                self.valid_lens_ctx,
                [self.s_test, self.f_test],
                *unbatch_args,
            )
            return PermutedSpatialData(None, *args)
        args = unbatch_BLD(
            [self.x_ctx, self.s_ctx, self.f_ctx],
            self.valid_lens_ctx,
            [self.x_test, self.s_test, self.f_test],
            *unbatch_args,
        )
        return PermutedSpatialData(*args)

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


@dataclass(frozen=True)
class SparseBatchedSpatialData(Batch):
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

# class SpatiotemporalTask(Enum):
#     Forecast = "forecast"
#     Interpolate = "interpolate"


# @dataclass(frozen=True)
# class SpatiotemporalData:
#     """A `SpatiotemporalData` container.

#     This class is intended for datasets with a spatial dimension, `s`, a
#     temporal dimension, `t`, optional fixed effects per time step per location, `x`,
#     and the functional output associated with each time step and location.

#     .. note::
#         Unlike the other `Data` classes, the batch dimension here is the time, `T`.

#     .. warning::
#         This container assumes that that timesteps are ordered from least to
#         most recent.
#     """

#     x: Optional[jax.Array]  # [T, [S]+, D_x] or [T, 1, D_x] or [1, 1, D_x] or None
#     s: jax.Array  # [T, [S]+, D_s]
#     t: jax.Array  # [T, [S]+, 1] or [T, 1, 1]
#     f: jax.Array  # [T, [S]+, D_f]

#     def to_batch(
#         self,
#         rng: jax.Array,
#         task: SpatiotemporalTask,
#         fixed_t_step: int,
#         num_t_per_element: int,
#         num_ctx_min_per_t: int,
#         num_ctx_max_per_t: int,
#         independent: bool,
#         num_test: int,
#         batch_size: int,
#         include_inv_permute_idx: bool = False,
#     ):
#         """Creates a `Batch` from this `SpatiotemporalData`.

#         Args:
#             rng: A PRNG.
#             task: A `SpatiotemporalTask` type, e.g. interpolate or forecast.
#             fixed_t_step: If greater than 0, use timesteps separated by a fixed
#                 step size along the leading axis, `T`, e.g. a `fixed_t_step=2`
#                 would yield indices 0, 2, 4, 6, ... along the `T` axis. Note
#                 that this likely only makes sense when the elements in the
#                 leading axis, `T`, are separated by a fixed time interval, e.g.
#                 t[0]=1.0, t[1]=2.0, t[2]=3.0, etc. If the time steps along
#                 the leading axis are real numbers with irregular gaps, e.g.
#                 t[0]=1.23, t[1]=5.7, t[3]=20.8, etc, this will likely have an
#                 unintended effect. In summary, `fixed_t_step` works with fixed
#                 indices along `T`, not with the actual times they represent.
#             num_t_per_element: Number of time steps to include in each batch
#                 element.
#             num_ctx_min_per_t: Minimum number of context points per time step.
#             num_ctx_max_per_t: Maximum number of context points per time step.
#             independent: Whether the subset of context points for each time step
#                 should be selected independently.
#             num_test: Number of test points for target time step.
#             batch_size: Number of elements in the generated batch.
#             include_inv_permute_idx: Whether to include the `inv_permute_idx`,
#                 which enables easily mapping context points back to their
#                 original times and locations in the dataset. This is useful for
#                 callbacks and plotting functions.
#         Returns:
#             A `Batch`.
#         """
#         return _spatiotemporal_data_to_batch(
#             rng,
#             self.x,
#             self.s,
#             self.t,
#             self.f,
#             task,
#             fixed_t_step,
#             num_t_per_element,
#             num_ctx_min_per_t,
#             num_ctx_max_per_t,
#             independent,
#             num_test,
#             batch_size,
#             include_inv_permute_idx,
#         )


# # register with jax to use in compiled functions
# jax.tree_util.register_pytree_node(
#     SpatiotemporalData,
#     lambda d: ((d.x, d.s, d.t, d.f), None),
#     lambda _aux, children: SpatiotemporalData(*children),
# )


# @partial(
#     jit,
#     static_argnames=(
#         "task",
#         "fixed_t_step",
#         "num_t_per_element",
#         "num_ctx_min_per_t",
#         "num_ctx_max_per_t",
#         "independent",
#         "num_test",
#         "batch_size",
#         "include_inv_permute_idx",
#     ),
# )
# def _spatiotemporal_data_to_batch(
#     rng: jax.Array,
#     x: Optional[jax.Array],  # [T, [S]+, D_x] or [T, 1, D_x] or [1, 1, D_x] or None
#     s: jax.Array,  # [T, [S]+, D_s]
#     t: jax.Array,  # [T, [S]+, 1] or [T, 1, 1]
#     f: jax.Array,  # [T, [S]+, D_f]
#     task: SpatiotemporalTask,
#     fixed_t_step: int,
#     num_t_per_element: int,
#     num_ctx_min_per_t: int,
#     num_ctx_max_per_t: int,
#     independent: bool,
#     num_test: int,
#     batch_size: int = 4,
#     include_inv_permute_idx: bool = False,
# ):
#     B, T, T_e = batch_size, s.shape[0], num_t_per_element
#     has_x = x is not None
#     flatten_spatial_dims = lambda v: v.reshape(T, -1, v.shape[-1])
#     x = flatten_spatial_dims(x) if has_x else None
#     s = flatten_spatial_dims(s)
#     t = flatten_spatial_dims(t)
#     f = flatten_spatial_dims(f)
#     L = s.shape[1]
#     num_test = num_test or L
#     x = jnp.broadcast_to(x, (T, L, x.shape[-1])) if has_x else None
#     t = jnp.broadcast_to(t, (T, L, 1))
#     rng_v, rng_t, rng = random.split(rng, 3)
#     valid_lens_ctx = random.randint(
#         rng_v, (B, T_e - 1), num_ctx_min_per_t, num_ctx_max_per_t
#     )
#     valid_lens_test = jnp.repeat(num_test, T)
#     if fixed_t_step < 0:  # use random time steps
#         rng_ts = random.split(rng_t, B)
#         ts = vmap(lambda rng: random.choice(rng, T, (T_e,), replace=False))(rng_ts)
#         ts = jnp.sort(ts, axis=1)
#     else:
#         rng_ts = random.split(rng_t, B)
#         choices = jnp.arange(fixed_t_step * (num_t_per_element - 1), T)
#         last_ts = random.choice(rng, choices, (B,), replace=False)
#         prev_ts = fixed_t_step * jnp.arange(num_t_per_element - 1, -1, -1)
#         ts = last_ts[:, None] - prev_ts
#     # last step if forecast, else median for interpolate
#     test_i = -1 if task is SpatiotemporalTask.Forecast else num_t_per_element // 2 + 1
#     # get indices of time steps for batches
#     t_test_i = ts[:, [test_i]]
#     t_ctx_i = jnp.concat([ts[:, :test_i], ts[:, test_i + 1 :]], axis=1)
#     # shapes: *_ctx: [B, T_e-1, [S]+, D_*], *_test: [B, 1, [S]+, D_*]
#     x_ctx, x_test = (x[t_ctx_i], x[t_test_i]) if has_x else (None, None)
#     s_ctx, s_test = s[t_ctx_i], s[t_test_i]
#     t_ctx, t_test = t[t_ctx_i], t[t_test_i]
#     f_ctx, f_test = f[t_ctx_i], f[t_test_i]
#     rngs = random.split(rng, T_e) if independent else jnp.repeat(rng, T_e)
#     # TODO:
#     # permute each T_e-1 ctx
#     # permute and pack test locs
#     # pack the context into a single array
#     x_ctx, x_test, _ = vbatch(rngs, x) if has_x else (None, None, None)
#     s_ctx, s_test, _ = vbatch(rngs, s)
#     t_ctx, t_test, _ = vbatch(rngs, t)
#     f_ctx, f_test, inv_permute_idx = vbatch(rngs, f)
#     return Batch(
#         x_ctx,
#         s_ctx,
#         t_ctx,
#         f_ctx,
#         valid_lens_ctx,
#         x_test,
#         s_test,
#         t_test,
#         f_test,
#         valid_lens_test,
#         # no need to store this if its not used
#         inv_permute_idx if include_inv_permute_idx else None,
#     )
