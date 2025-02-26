import jax.numpy as jnp
from jax import random
from sps.utils import build_grid

from dl4bi.meta_learning.data.spatial import SpatialData
from dl4bi.meta_learning.data.tabular import TabularData
from dl4bi.meta_learning.data.temporal import TemporalData


def test_tabular_data():
    rng = random.key(42)
    B, L, D_x, D_f = 4, 37, 8, 1
    num_ctx_min, num_ctx_max, num_test = 3, 10, 20
    x_shape = (B, L, D_x)
    f_shape = (B, L, D_f)
    x_ctx_shape = (B, num_ctx_max, D_x)
    f_ctx_shape = (B, num_ctx_max, D_f)
    x_test_shape = (B, num_test, D_x)
    f_test_shape = (B, num_test, D_f)
    x = random.normal(rng, x_shape)
    f = random.normal(rng, f_shape)
    # test basic instantiation
    d = TabularData(x, f)
    assert d.x.shape == x_shape
    assert d.f.shape == f_shape
    # test permute and inv_permute
    dp = d.permute(rng)
    _d = dp.inv_permute()
    assert d.eq_shapes(dp)
    assert d == _d
    assert dp.inv_permute_idx.shape == (L,)
    # test permute and inv_permute with independent permutations
    dp = d.permute(rng, independent=True)
    _d = dp.inv_permute()
    assert d.eq_shapes(dp)
    assert d == _d
    assert dp.inv_permute_idx.shape == (B, L)
    # test batching where test includes context
    test_includes_ctx = True
    b = dp.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx.shape == x_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.valid_lens_ctx.shape == (B,)
    assert b.x_test.shape == x_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.valid_lens_test.shape == (B,)
    assert (b.x_ctx[:, :num_ctx_max] == b.x_test[:, :num_ctx_max]).all()
    assert (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test batching where test does not include context
    test_includes_ctx = False
    b = dp.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx.shape == x_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.valid_lens_ctx.shape == (B,)
    assert b.x_test.shape == x_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.valid_lens_test.shape == (B,)
    assert not (b.x_ctx[:, :num_ctx_max] == b.x_test[:, :num_ctx_max]).all()
    assert not (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test unbatching
    _dp = b.unbatch()
    _d = _dp.inv_permute()
    assert dp.eq_shapes(_dp)
    assert (dp.inv_permute_idx == dp.inv_permute_idx).all()
    assert d.eq_shapes(_d)


def test_spatial_data_with_x():
    rng = random.key(42)
    B, S, D_x, D_s, D_f = 4, 16, 8, 2, 1
    num_ctx_min, num_ctx_max, num_test = 3, 10, 20
    x_shape = (B, *[S] * D_s, D_x)
    s_shape = (B, *[S] * D_s, D_s)
    f_shape = (B, *[S] * D_s, D_f)
    x_ctx_shape = (B, num_ctx_max, D_x)
    s_ctx_shape = (B, num_ctx_max, D_s)
    f_ctx_shape = (B, num_ctx_max, D_f)
    x_test_shape = (B, num_test, D_x)
    s_test_shape = (B, num_test, D_s)
    f_test_shape = (B, num_test, D_f)
    x = random.normal(rng, x_shape)
    f = random.normal(rng, f_shape)
    s = build_grid([{"start": -2, "stop": 2, "num": S}] * D_s)
    s = jnp.repeat(s[None, ...], B, axis=0)
    # test basic instantiation
    d = SpatialData(x, s, f)
    assert d.x.shape == x_shape
    assert d.s.shape == s_shape
    assert d.f.shape == f_shape
    # test permute and inv_permute
    dp = d.permute(rng)
    _d = dp.inv_permute()
    assert d.eq_shapes(dp)
    assert d == _d
    assert dp.inv_permute_idx.shape == (S**D_s,)
    # test permute and inv_permute with independent permutations
    dp = d.permute(rng, independent=True)
    _d = dp.inv_permute()
    assert d.eq_shapes(dp)
    assert d == _d
    assert dp.inv_permute_idx.shape == (B, S**D_s)
    # test batching where test includes context
    test_includes_ctx = True
    b = dp.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx.shape == x_ctx_shape
    assert b.s_ctx.shape == s_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.valid_lens_ctx.shape == (B,)
    assert b.x_test.shape == x_test_shape
    assert b.s_test.shape == s_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.valid_lens_test.shape == (B,)
    assert (b.x_ctx[:, :num_ctx_max] == b.x_test[:, :num_ctx_max]).all()
    assert (b.s_ctx[:, :num_ctx_max] == b.s_test[:, :num_ctx_max]).all()
    assert (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test batching where test does not include context
    test_includes_ctx = False
    b = dp.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx.shape == x_ctx_shape
    assert b.s_ctx.shape == s_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.valid_lens_ctx.shape == (B,)
    assert b.x_test.shape == x_test_shape
    assert b.s_test.shape == s_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.valid_lens_test.shape == (B,)
    assert not (b.x_ctx[:, :num_ctx_max] == b.x_test[:, :num_ctx_max]).all()
    assert not (b.s_ctx[:, :num_ctx_max] == b.s_test[:, :num_ctx_max]).all()
    assert not (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test unbatching
    _dp = b.unbatch()
    _d = _dp.inv_permute()
    assert dp.eq_shapes(_dp)
    assert (dp.inv_permute_idx == dp.inv_permute_idx).all()
    assert d.eq_shapes(_d)


def test_spatial_data_broadcast_x():
    rng = random.key(42)
    B, S, D_x, D_s, D_f = 4, 16, 8, 2, 1
    num_ctx_min, num_ctx_max, num_test = 3, 10, 20
    x_shape = (B, D_x)
    s_shape = (B, *[S] * D_s, D_s)
    f_shape = (B, *[S] * D_s, D_f)
    x_ctx_shape = (B, num_ctx_max, D_x)
    s_ctx_shape = (B, num_ctx_max, D_s)
    f_ctx_shape = (B, num_ctx_max, D_f)
    x_test_shape = (B, num_test, D_x)
    s_test_shape = (B, num_test, D_s)
    f_test_shape = (B, num_test, D_f)
    x = random.normal(rng, x_shape)
    f = random.normal(rng, f_shape)
    s = build_grid([{"start": -2, "stop": 2, "num": S}] * D_s)
    s = jnp.repeat(s[None, ...], B, axis=0)
    # test basic instantiation
    d = SpatialData(x, s, f)
    assert d.x.shape == x_shape
    assert d.s.shape == s_shape
    assert d.f.shape == f_shape
    # test permute and inv_permute
    dp = d.permute(rng)
    _d = dp.inv_permute()
    assert d.eq_shapes(dp)
    assert d == _d
    assert dp.inv_permute_idx.shape == (S**D_s,)
    # test permute and inv_permute with independent permutations
    dp = d.permute(rng, independent=True)
    _d = dp.inv_permute()
    assert d.eq_shapes(dp)
    assert d == _d
    assert dp.inv_permute_idx.shape == (B, S**D_s)
    # test batching where test includes context
    test_includes_ctx = True
    b = dp.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx.shape == x_ctx_shape
    assert b.s_ctx.shape == s_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.valid_lens_ctx.shape == (B,)
    assert b.x_test.shape == x_test_shape
    assert b.s_test.shape == s_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.valid_lens_test.shape == (B,)
    assert (b.x_ctx[:, :num_ctx_max] == b.x_test[:, :num_ctx_max]).all()
    assert (b.s_ctx[:, :num_ctx_max] == b.s_test[:, :num_ctx_max]).all()
    assert (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test batching where test does not include context
    test_includes_ctx = False
    b = dp.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx.shape == x_ctx_shape
    assert b.s_ctx.shape == s_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.valid_lens_ctx.shape == (B,)
    assert b.x_test.shape == x_test_shape
    assert b.s_test.shape == s_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.valid_lens_test.shape == (B,)
    assert not (b.s_ctx[:, :num_ctx_max] == b.s_test[:, :num_ctx_max]).all()
    assert not (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test unbatching
    _dp = b.unbatch()
    _d = _dp.inv_permute()
    assert dp.eq_shapes(_dp)
    assert (dp.inv_permute_idx == dp.inv_permute_idx).all()
    assert d.eq_shapes(_d)


def test_spatial_data_without_x():
    rng = random.key(42)
    B, S, D_s, D_f = 4, 16, 2, 1
    num_ctx_min, num_ctx_max, num_test = 3, 10, 20
    s_shape = (B, *[S] * D_s, D_s)
    f_shape = (B, *[S] * D_s, D_f)
    s_ctx_shape = (B, num_ctx_max, D_s)
    f_ctx_shape = (B, num_ctx_max, D_f)
    s_test_shape = (B, num_test, D_s)
    f_test_shape = (B, num_test, D_f)
    x = None
    f = random.normal(rng, f_shape)
    s = build_grid([{"start": -2, "stop": 2, "num": S}] * D_s)
    s = jnp.repeat(s[None, ...], B, axis=0)
    # test basic instantiation
    d = SpatialData(x, s, f)
    assert d.x is None
    assert d.s.shape == s_shape
    assert d.f.shape == f_shape
    # test permute and inv_permute
    dp = d.permute(rng)
    _d = dp.inv_permute()
    assert d.eq_shapes(dp)
    assert d == _d
    assert dp.inv_permute_idx.shape == (S**D_s,)
    # test permute and inv_permute with independent permutations
    dp = d.permute(rng, independent=True)
    _d = dp.inv_permute()
    assert d.eq_shapes(dp)
    assert d == _d
    assert dp.inv_permute_idx.shape == (B, S**D_s)
    # test batching where test includes context
    test_includes_ctx = True
    b = dp.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx is None
    assert b.s_ctx.shape == s_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.valid_lens_ctx.shape == (B,)
    assert b.x_test is None
    assert b.s_test.shape == s_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.valid_lens_test.shape == (B,)
    assert (b.s_ctx[:, :num_ctx_max] == b.s_test[:, :num_ctx_max]).all()
    assert (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test batching where test does not include context
    test_includes_ctx = False
    b = dp.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx is None
    assert b.s_ctx.shape == s_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.valid_lens_ctx.shape == (B,)
    assert b.x_test is None
    assert b.s_test.shape == s_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.valid_lens_test.shape == (B,)
    assert not (b.s_ctx[:, :num_ctx_max] == b.s_test[:, :num_ctx_max]).all()
    assert not (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test unbatching
    _dp = b.unbatch()
    _d = _dp.inv_permute()
    assert dp.eq_shapes(_dp)
    assert (dp.inv_permute_idx == dp.inv_permute_idx).all()
    assert d.eq_shapes(_d)


def test_temporal_data_with_x():
    rng = random.key(42)
    B, T, D_x, D_f = 4, 64, 8, 2
    num_ctx_min, num_ctx_max, num_test = 3, 10, 20
    x_shape = (B, T, D_x)
    t_shape = (B, T)
    f_shape = (B, T, D_f)
    x_ctx_shape = (B, num_ctx_max, D_x)
    t_ctx_shape = (B, num_ctx_max, 1)
    f_ctx_shape = (B, num_ctx_max, D_f)
    x_test_shape = (B, num_test, D_x)
    t_test_shape = (B, num_test, 1)
    f_test_shape = (B, num_test, D_f)
    x = random.normal(rng, x_shape)
    f = random.normal(rng, f_shape)
    t = jnp.repeat(jnp.arange(T)[None, ...], B, axis=0)
    # test basic instantiation
    d = TemporalData(x, t, f)
    assert d.x.shape == x_shape
    assert d.t.shape == t_shape
    assert d.f.shape == f_shape
    # test permute and inv_permute
    dp = d.permute(rng)
    _d = dp.inv_permute()
    assert d.eq_shapes(dp)
    assert d == _d
    assert dp.inv_permute_idx.shape == (T,)
    # test permute and inv_permute with independent permutations
    dp = d.permute(rng, independent=True)
    _d = dp.inv_permute()
    assert d.eq_shapes(dp)
    assert d == _d
    assert dp.inv_permute_idx.shape == (B, T)
    # test batching where test includes context
    test_includet_ctx = True
    b = dp.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includet_ctx)
    assert b.x_ctx.shape == x_ctx_shape
    assert b.t_ctx.shape == t_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.valid_lens_ctx.shape == (B,)
    assert b.x_test.shape == x_test_shape
    assert b.t_test.shape == t_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.valid_lens_test.shape == (B,)
    assert (b.x_ctx[:, :num_ctx_max] == b.x_test[:, :num_ctx_max]).all()
    assert (b.t_ctx[:, :num_ctx_max] == b.t_test[:, :num_ctx_max]).all()
    assert (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test batching where test does not include context
    test_includet_ctx = False
    b = dp.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includet_ctx)
    assert b.x_ctx.shape == x_ctx_shape
    assert b.t_ctx.shape == t_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.valid_lens_ctx.shape == (B,)
    assert b.x_test.shape == x_test_shape
    assert b.t_test.shape == t_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.valid_lens_test.shape == (B,)
    assert not (b.x_ctx[:, :num_ctx_max] == b.x_test[:, :num_ctx_max]).all()
    assert not (b.t_ctx[:, :num_ctx_max] == b.t_test[:, :num_ctx_max]).all()
    assert not (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test unbatching
    _dp = b.unbatch()
    _d = _dp.inv_permute()
    assert dp.eq_shapes(_dp)
    assert (dp.inv_permute_idx == dp.inv_permute_idx).all()
    assert d.eq_shapes(_d)


def test_temporal_data_broadcast_x():
    rng = random.key(42)
    B, T, D_x, D_f = 4, 64, 8, 2
    num_ctx_min, num_ctx_max, num_test = 3, 10, 20
    x_shape = (B, D_x)
    t_shape = (B, T)
    f_shape = (B, T, D_f)
    x_ctx_shape = (B, num_ctx_max, D_x)
    t_ctx_shape = (B, num_ctx_max, 1)
    f_ctx_shape = (B, num_ctx_max, D_f)
    x_test_shape = (B, num_test, D_x)
    t_test_shape = (B, num_test, 1)
    f_test_shape = (B, num_test, D_f)
    x = random.normal(rng, x_shape)
    f = random.normal(rng, f_shape)
    t = jnp.repeat(jnp.arange(T)[None, ...], B, axis=0)
    # test basic instantiation
    d = TemporalData(x, t, f)
    assert d.x.shape == x_shape
    assert d.t.shape == t_shape
    assert d.f.shape == f_shape
    # test permute and inv_permute
    dp = d.permute(rng)
    _d = dp.inv_permute()
    assert d.eq_shapes(dp)
    assert d == _d
    assert dp.inv_permute_idx.shape == (T,)
    # test permute and inv_permute with independent permutations
    dp = d.permute(rng, independent=True)
    _d = dp.inv_permute()
    assert d.eq_shapes(dp)
    assert d == _d
    assert dp.inv_permute_idx.shape == (B, T)
    # test batching where test includes context
    test_includet_ctx = True
    b = dp.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includet_ctx)
    assert b.x_ctx.shape == x_ctx_shape
    assert b.t_ctx.shape == t_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.valid_lens_ctx.shape == (B,)
    assert b.x_test.shape == x_test_shape
    assert b.t_test.shape == t_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.valid_lens_test.shape == (B,)
    assert (b.x_ctx[:, :num_ctx_max] == b.x_test[:, :num_ctx_max]).all()
    assert (b.t_ctx[:, :num_ctx_max] == b.t_test[:, :num_ctx_max]).all()
    assert (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test batching where test does not include context
    test_includet_ctx = False
    b = dp.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includet_ctx)
    assert b.x_ctx.shape == x_ctx_shape
    assert b.t_ctx.shape == t_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.valid_lens_ctx.shape == (B,)
    assert b.x_test.shape == x_test_shape
    assert b.t_test.shape == t_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.valid_lens_test.shape == (B,)
    assert not (b.t_ctx[:, :num_ctx_max] == b.t_test[:, :num_ctx_max]).all()
    assert not (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test unbatching
    _dp = b.unbatch()
    _d = _dp.inv_permute()
    assert dp.eq_shapes(_dp)
    assert (dp.inv_permute_idx == dp.inv_permute_idx).all()
    assert d.eq_shapes(_d)


def test_temporal_data_without_x():
    rng = random.key(42)
    B, T, D_f = 4, 64, 2
    num_ctx_min, num_ctx_max, num_test = 3, 10, 20
    t_shape = (B, T)
    f_shape = (B, T, D_f)
    t_ctx_shape = (B, num_ctx_max, 1)
    f_ctx_shape = (B, num_ctx_max, D_f)
    t_test_shape = (B, num_test, 1)
    f_test_shape = (B, num_test, D_f)
    x = None
    f = random.normal(rng, f_shape)
    t = jnp.repeat(jnp.arange(T)[None, ...], B, axis=0)
    # test basic instantiation
    d = TemporalData(x, t, f)
    assert d.x is None
    assert d.t.shape == t_shape
    assert d.f.shape == f_shape
    # test permute and inv_permute
    dp = d.permute(rng)
    _d = dp.inv_permute()
    assert d.eq_shapes(dp)
    assert d == _d
    assert dp.inv_permute_idx.shape == (T,)
    # test permute and inv_permute with independent permutations
    dp = d.permute(rng, independent=True)
    _d = dp.inv_permute()
    assert d.eq_shapes(dp)
    assert d == _d
    assert dp.inv_permute_idx.shape == (B, T)
    # test batching where test includes context
    test_includet_ctx = True
    b = dp.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includet_ctx)
    assert b.x_ctx is None
    assert b.t_ctx.shape == t_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.valid_lens_ctx.shape == (B,)
    assert b.x_test is None
    assert b.t_test.shape == t_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.valid_lens_test.shape == (B,)
    assert (b.t_ctx[:, :num_ctx_max] == b.t_test[:, :num_ctx_max]).all()
    assert (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test batching where test does not include context
    test_includet_ctx = False
    b = dp.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includet_ctx)
    assert b.x_ctx is None
    assert b.t_ctx.shape == t_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.valid_lens_ctx.shape == (B,)
    assert b.x_test is None
    assert b.t_test.shape == t_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.valid_lens_test.shape == (B,)
    assert not (b.t_ctx[:, :num_ctx_max] == b.t_test[:, :num_ctx_max]).all()
    assert not (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test unbatching
    _dp = b.unbatch()
    _d = _dp.inv_permute()
    assert dp.eq_shapes(_dp)
    assert (dp.inv_permute_idx == dp.inv_permute_idx).all()
    assert d.eq_shapes(_d)
