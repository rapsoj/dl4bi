from dl4bi.regression.d_linear import DLinear
from jax import random


def test_d_linear():
    B, L, N = 32, 365 * 2, 30
    rng = random.key(42)
    rng_data, rng_init = random.split(rng)
    x = random.normal(rng_data, (B, L))
    for seasonal_lags in [[], [7], [365], [7, 365]]:
        m = DLinear(N, seasonal_lags)
        y, _params = m.init_with_output(rng_init, x)
        assert y.shape == (B, N)
