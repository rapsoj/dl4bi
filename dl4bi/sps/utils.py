from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, random
from jax.typing import ArrayLike


def build_grid(
    axes: Sequence[dict[str, jax.Array | float]] = [
        {"start": 0, "stop": 1, "num": 128}
    ],
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Builds a grid of shape `[..., D]` along the axes using `jnp.linspace`.

    Args:
        axes: A list of dicts, each with keys `start`, `stop`, and `num`, which
            are passed to `jnp.linspace`.

    Returns:
        A mesh grid across those axes.
    """
    pts = [jnp.linspace(**axis, dtype=dtype) for axis in axes]
    return jnp.stack(jnp.meshgrid(*pts, indexing="ij"), axis=-1)


def scale_grid(grid: ArrayLike, factor: int) -> jax.Array:
    """Scales the `grid` of shape `[..., D]` by `factor` along all axes.

    Args:
        grid: A mesh grid.
        factor: A factor by which to scale each dimension of the grid.

    Returns:
        A scaled grid.
    """
    axes = [
        jnp.linspace(grid[..., dim].min(), grid[..., dim].max(), int(n * factor))
        for dim, n in enumerate(grid.shape[:-1])
    ]
    return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)


def random_subgrid(
    rng: jax.Array,
    axes: Sequence[dict[str, float]] = [{"start": 0, "stop": 1, "num": 32}] * 2,
    min_axes_pct: float = 0.05,
    max_axes_pct: float = 1.0,
):
    """Create a random subgrid from `axes` at the same resolution.

    .. warning::
        This method assumes that the `start` points always comes before the
        `stop` point on the real number line.
    """
    D = len(axes)
    rng_width, rng_shift = random.split(rng)
    u_width = random.uniform(rng_width, (1,), minval=min_axes_pct, maxval=max_axes_pct)
    u_width = u_width[0]
    u_corner = random.uniform(rng_shift, (D,), maxval=1 - u_width)  # bottom left
    u_center = jnp.array([0.5] * D)
    lower_left = jnp.array([d["start"] for d in axes])
    upper_right = jnp.array([d["stop"] for d in axes])
    scale = upper_right - lower_left
    center = (upper_right + lower_left) / 2
    corner = (u_corner - u_center + center) * scale
    width = u_width * scale
    return build_grid(
        [
            {"start": corner[i], "stop": corner[i] + width[i], "num": axes[i]["num"]}
            for i in range(D)
        ]
    )


@partial(jit, static_argnames=("width",))
def inv_dist_sq_kernel(width: int = 7):
    center = width // 2
    x = y = jnp.arange(width) - center
    xx, yy = jnp.meshgrid(x, y)
    dist_sq = jnp.float32(xx**2 + yy**2)
    return 1 / dist_sq.at[center, center].set(jnp.inf)
