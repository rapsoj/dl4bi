"""
Based on reference implementation here: https://krasserm.github.io/2018/03/19/gaussian-processes/
"""

from collections.abc import Callable

import numpy as np
from jax.typing import ArrayLike
from numpy.linalg import cholesky, det, inv
from scipy.linalg import solve_triangular
from scipy.optimize import minimize


def find_gp_mle(s: ArrayLike, f: ArrayLike, kernel: Callable, jitter: float = 1e-5):
    res = minimize(
        _nll_fn_stable(s, f, kernel, jitter),
        [2.0, 2.0],
        bounds=((1e-5, None), (1e-5, None)),
        method="L-BFGS-B",
        options=dict(ftol=0.01, gtol=1e-10),
    )
    print(res)
    return res.x  # (ls, var)


def _nll_fn_stable(s: ArrayLike, f: ArrayLike, kernel: Callable, jitter: float):
    N, D = s.size // s.shape[-1], s.shape[-1]
    s = s.reshape(-1, D)
    f = f.reshape(-1)

    def nll(theta):
        var, ls = theta
        K = kernel(s, s, var, ls) + jitter * np.eye(N)
        L = cholesky(K)
        S1 = solve_triangular(L, f, lower=True)
        S2 = solve_triangular(L.T, S1, lower=False)
        # TODO(danj): ignore constant terms
        return np.sum(np.log(np.diag(L))) + 0.5 * f @ S2 + 0.5 * N * np.log(2 * np.pi)

    return nll


def _nll_fn(s: ArrayLike, f: ArrayLike, kernel: Callable, jitter: float):
    N, D = s.size // s.shape[-1], s.shape[-1]
    s = s.reshape(-1, D)
    f = f.reshape(-1)

    def nll(theta):
        var, ls = theta
        K = kernel(s, s, var, ls) + jitter * np.eye(N)
        return 0.5 * np.log(det(K)) + 0.5 * f @ inv(K) @ f + 0.5 * N * np.log(2 * np.pi)

    return nll
