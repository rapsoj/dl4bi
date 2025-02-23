from dataclasses import asdict, dataclass, fields, replace
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import jit, random


@dataclass(frozen=True)
class BatchElement:
    """Same as a batch but without the leading batch dim."""

    x_ctx: Optional[jax.Array] = None
    s_ctx: Optional[jax.Array] = None
    f_ctx: Optional[jax.Array] = None
    valid_lens_ctx: Optional[jax.Array] = None
    x_test: Optional[jax.Array] = None
    s_test: Optional[jax.Array] = None
    f_test: Optional[jax.Array] = None
    valid_lens_test: Optional[jax.Array] = None
    inv_permute_idx: Optional[jax.Array] = None

    def update(self, **kwargs):
        """Returns a new batch with updated attributes."""
        return replace(self, **kwargs)

    def __iter__(self):
        """Allows you to use **batch to expand as kwargs."""
        yield from asdict(self).items()

    def __len__(self):
        for field in fields(self):
            if field.name.endswith(("ctx", "test")):
                x = getattr(self, field.name)
                if isinstance(x, jax.Array):
                    return x.shape[0]
        return None


@dataclass(frozen=True)
class Batch:
    """A batch."""

    x_ctx: Optional[jax.Array] = None  # [B, L, D_x]
    s_ctx: Optional[jax.Array] = None  # [B, L, D_s]
    f_ctx: Optional[jax.Array] = None  # [B, L, D_f]
    valid_lens_ctx: Optional[jax.Array] = None  # [B, L]
    x_test: Optional[jax.Array] = None  # [B, L, D_x]
    s_test: Optional[jax.Array] = None  # [B, L, D_s]
    f_test: Optional[jax.Array] = None  # [B, L, D_f]
    valid_lens_test: Optional[jax.Array] = None  # [B, L]
    inv_permute_idx: Optional[jax.Array] = None  # [B, T, L] or [B, L] or [L]
    time_start_idx: Optional[jax.Array] = None  # [B, T]

    def update(self, **kwargs):
        """Returns a new batch with updated attributes."""
        return replace(self, **kwargs)

    def __iter__(self):
        """Allows you to use **batch to expand as kwargs."""
        yield from asdict(self).items()

    def __getitem__(self, i: int):
        d = {}
        for k, v in self.items():
            if isinstance(v, jax.Array):
                if k.endswith(("ctx", "test")):
                    v = v[i]  # [B] or [B, L, D]
                elif k == "inv_permute_idx":
                    # inv_permute_idx shape meaning:
                    # [L]: A single permutation for all batch elements
                    # [B, L]: A separate permutation for each batch element
                    # [B, T, L]: A separate permutation for each batch element and timestep
                    if v.ndim > 1:
                        v = v[i]
            d[k] = v
        return BatchElement(**d)

    def __len__(self):
        for field in fields(self):
            if field.name.endswith(("ctx", "test")):
                x = getattr(self, field.name)
                if isinstance(x, jax.Array):
                    return x.shape[0]
        return None


@partial(
    jit,
    static_argnames=(
        "s_min_ctx",
        "s_max_ctx",
        "s_min_test",
        "s_max_test",
        "batch_size",
        "independent",
        "s_test_includes_s_ctx",
    ),
)
def spatial_data_to_batch(
    rng: jax.Array,
    x: Optional[jax.Array],  # [B, [S]+, D_x] or None
    s: jax.Array,  # [B, [S]+]
    f: jax.Array,  # [B, [S]+, D_f]
    s_min_ctx: Optional[int] = None,
    s_max_ctx: Optional[int] = None,
    s_min_test: Optional[int] = None,
    s_max_test: Optional[int] = None,
    batch_size: int = 4,
    independent: bool = False,
    s_test_includes_s_ctx: bool = True,
):
    B = batch_size
    flatten_spatial_dims = lambda x: x.reshape(B, -1, x.shape[-1])
    if x is not None:
        x = flatten_spatial_dims(x)
    s, f = flatten_spatial_dims(s), flatten_spatial_dims(f)
    L = s.shape[1]
    Nc_max = s_max_ctx or L
    Nc_min = s_min_ctx or Nc_max
    valid_lens_ctx = None
    if Nc_min != Nc_max:
        rng_valid, rng = random.split(rng)
        valid_lens_ctx = random.randint(rng_valid, (B,), Nc_min, Nc_max)
    if independent:
        inv_perms, s_ctxs, f_ctxs, s_tests, f_tests = [], [], []
        for i in range(B):
            rng_i, rng = random.split(rng)
            permute_idx = random.choice(rng_i, L, (L,), replace=False)
            inv_perms += jnp.argsort(permute_idx)
            s_i_permuted = s[i, permute_idx, :]
            f_i_permuted = f[i, permute_idx, :]
            s_ctx = s_i_permuted[:, :Nc_max]
            f_ctx = f_i_permuted[:, :Nc_max]

    pass


@partial(
    jit,
    static_argnames=(
        "s_min_ctx",
        "s_max_ctx",
        "t_indep_s_ctx",
        "t_fixed_interval",
        "t_target_is_last",
        "batch_size",
    ),
)
def spatiotemporal_data_to_batch(
    rng: jax.Array,
    x: Optional[jax.Array],  # [T, [S]+, D_x] or [[S]+, D_x] or None
    s: jax.Array,  # [T, [S]+] or [S]+
    t: jax.Array,  # [T]
    f: jax.Array,  # [T, [S]+, D_f]
    s_min_ctx: Optional[int] = None,
    s_max_ctx: Optional[int] = None,
    t_num_ctx: int = 4,
    t_indep_s_ctx: bool = False,
    t_fixed_interval: Optional[int] = None,
    t_target_is_last: bool = True,
    batch_size: int = 4,
):
    B = batch_size
    rng_ts, rng = random.split(rng)
    if t_target_is_last:  # last time step is the target time step
        rng_lasts, rng_ts = random.split(rng)
        t_lasts = random.randint(rng_lasts, (B,), t.min() + t_num_ctx, t.max())
        ts = []
        for t_last in t_lasts:
            rng_t, rng_ts = random.split(rng_ts)
            ts += [random.randint(rng_t, ())]

    pass
