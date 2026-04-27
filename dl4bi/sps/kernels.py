import jax.numpy as jnp
from jax import jit, vmap
from jax.typing import ArrayLike


@jit
def _prepare_dims(x: ArrayLike, y: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """Prepares dims for use in kernel functions.

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        `[N_x, D]` and `[N_y, D]` arrays.
    """
    if x.ndim == 1:
        x = x[:, jnp.newaxis]
    if y.ndim == 1:
        y = y[:, jnp.newaxis]
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1, y.shape[-1])
    return x, y


@jit
def l2_dist_sq(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    r"""Squared L2 distance between two [..., D] arrays.

    .. note::
        This is more numerically stable than the factorization
        trick: ||a + b||^2 = ||a||^2 + ||b||^2 - 2*||a||*||b||.

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        Matrix of all pairwise distances.
    """
    x, y = _prepare_dims(x, y)
    d = x[:, None, :] - y[None, :, :]
    return jnp.sum(d**2, axis=-1)


@jit
def l2_dist(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    r"""L2 distance between two [..., D] arrays.

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        Matrix of all pairwise distances.
    """
    x, y = _prepare_dims(x, y)
    d = x[:, None, :] - y[None, :, :]
    return jnp.linalg.norm(d, axis=-1)


@jit
def rbf(
    x: ArrayLike,
    y: ArrayLike,
    var: float,
    ls: float,
) -> ArrayLike:
    r"""Radial Basis kernel, aka Squared Exponential kernel.

    $K(x, y) = \text{var}\cdot\exp\left(-\frac{\lVert x-y\rVert^2}{2\text{ls}^2}\right)$

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        A covariance matrix.
    """
    return var * jnp.exp(-l2_dist_sq(x, y) / (2 * ls**2))


@jit
def periodic(
    x: ArrayLike,
    y: ArrayLike,
    var: float,
    ls: float,
    period: float = 0.5,  # 2 cycles on unit interval
) -> ArrayLike:
    r"""Periodic kernel.

    $K(x, y) = \text{var}\cdot\exp\left(-\frac{2\sin^2\frac{\left(\pi\lVert x-y\rVert\right)}{\text{period}}}{\text{ls}^2}\right)$

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        A covariance matrix.
    """
    x, y = _prepare_dims(x, y)
    return var * jnp.exp(-2 / ls**2 * jnp.sin(jnp.pi * jnp.abs(x - y.T) / period) ** 2)


@jit
def exponential(
    x: ArrayLike,
    y: ArrayLike,
    var: float,
    ls: float,
) -> ArrayLike:
    r"""Exponential kernel. Alias of Matern 1/2 kernel.

    $K(x, y) = \text{var}\cdot\left(-\frac{\lVert x-y\rVert}{\text{ls}}\right)$

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        A covariance matrix.
    """
    return matern_1_2(x, y, var, ls)


@jit
def matern_1_2(
    x: ArrayLike,
    y: ArrayLike,
    var: float,
    ls: float,
) -> ArrayLike:
    r"""Matern 1/2 kernel.

    $K(x, y) = \text{var}\cdot\left(-\frac{\lVert x-y\rVert}{\text{ls}}\right)$

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        A covariance matrix.
    """
    x, y = _prepare_dims(x, y)
    return var * jnp.exp(-l2_dist(x, y) / ls)


@jit
def matern_3_2(
    x: ArrayLike,
    y: ArrayLike,
    var: float,
    ls: float,
) -> ArrayLike:
    r"""Matern 3/2 kernel.

    $K(x, y) = \text{var}\cdot\left(1 + \frac{\sqrt{3}\lVert x-y\rVert}{\text{ls}}\right)\cdot\exp\left(-\frac{\sqrt{3}\lVert x-y\rVert}{\text{ls}}\right)$

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        A covariance matrix.
    """
    d = l2_dist(x, y)
    sqrt3 = 3.0 ** (1 / 2)
    return var * (1 + sqrt3 * d / ls) * jnp.exp(-sqrt3 * d / ls)


@jit
def matern_5_2(
    x: ArrayLike,
    y: ArrayLike,
    var: float,
    ls: float,
) -> ArrayLike:
    r"""Matern 5/2 kernel.

    $K(x, y) = \text{var}\cdot\left(1 + \frac{\sqrt{5}\lVert x-y\rVert}{\text{ls}} + \frac{5}{3}\cdot\frac{\lVert x-y\rVert^2}{\text{ls}^2}\right)\cdot\exp\left(-\frac{\sqrt{5}\lVert x-y\rVert}{\text{ls}}\right)$

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        A covariance matrix.
    """
    d = l2_dist(x, y)
    dsq = jnp.square(d)
    sqrt5 = jnp.sqrt(5.0)
    return var * (1 + sqrt5 * d / ls + 5 / 3 * dsq / ls**2) * jnp.exp(-sqrt5 * d / ls)


@jit
def great_circle_dist(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    r"""Great circle distance on a sphere between two [..., 2] arrays.

    Inputs are assumed to be pairs of (longitude, latitude) in degrees,
    outputs are also returned in degrees.

    Args:
        x: Input array of size `[..., 2]`.
        y: Input array of size `[..., 2]`.

    Returns:
        Matrix of all pairwise distances.
    """

    def d(x, y):
        x_lon, x_lat = x
        y_lon, y_lat = y
        x_lon, x_lat, y_lon, y_lat = map(jnp.deg2rad, (x_lon, x_lat, y_lon, y_lat))

        d_lon = jnp.abs(x_lon - y_lon)

        sin = jnp.sin
        cos = jnp.cos

        arc_length = jnp.atan2(
            jnp.sqrt(
                (cos(y_lat) * sin(d_lon)) ** 2
                + (cos(x_lat) * sin(y_lat) - sin(x_lat) * cos(y_lat) * cos(d_lon)) ** 2
            ),
            sin(x_lat) * sin(y_lat) + cos(x_lat) * cos(y_lat) * cos(d_lon),
        )

        return jnp.rad2deg(arc_length)

    assert x.shape[-1] == y.shape[-1] == 2, "Input arrays must be of shape [..., 2]"
    x, y = _prepare_dims(x, y)
    return vmap(vmap(d, in_axes=(None, 0)), in_axes=(0, None))(x, y)


@jit
def geo_exponential(
    x: ArrayLike,
    y: ArrayLike,
    var: float,
    ls: float,
) -> ArrayLike:
    r"""Geodesic exponential kernel, that is an exponential kernel with great circle distance.

    $K(x, y) = \text{var}\cdot\exp\left(-\frac{\lVert x-y\rVert}_{geo}{\text{ls}}\right)$

    Args:
        x: Input array of size `[..., 2]`.
        y: Input array of size `[..., 2]`.

    Returns:
        A covariance matrix.
    """
    return var * jnp.exp(-great_circle_dist(x, y) / ls)
