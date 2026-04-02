import jax.numpy as jnp
import jax.random as jr
import numpy as np
from tqdm import tqdm

from sps.kernels import (st_separable_rbf_matern_1_2,
                         st_nonseparable_gneiting,
                         st_nonseparable_gneiting_advected)

import matplotlib.pyplot as plt
import imageio


def sample_gp(kernel_fn, coords, key, jitter=1e-3):
    """
    coords: [N, 3] = (x, y, t)
    """
    K = kernel_fn(coords, coords)

    z = jr.normal(key, (K.shape[0],))

    # Stabilize
    K = K + jitter * jnp.eye(K.shape[0])
    K = 0.5 * (K + K.T) # force matrix to be exactly symmetric for Cholesky

    L = jnp.linalg.cholesky(K)

    z = jr.normal(key, (K.shape[0],))
    f = L @ z

    return f


def make_grid(nx=20, ny=20, nt=30):
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    t = jnp.linspace(0, 1, nt)

    xx, yy, tt = jnp.meshgrid(x, y, t, indexing="ij")

    coords = jnp.stack([
        xx.ravel(),
        yy.ravel(),
        tt.ravel()
    ], axis=-1)

    return coords, (nx, ny, nt)


def reshape_to_frames(f, shape):
    nx, ny, nt = shape
    return f.reshape(nx, ny, nt)


def save_gif(frames, filename, vmin=None, vmax=None):
    images = []

    for t in tqdm(range(frames.shape[-1]), f"Generating plots for {filename}"):
        fig, ax = plt.subplots()
        im = ax.imshow(np.array(frames[:, :, t]),
                       origin="lower",
                       cmap="viridis",
                       vmin=vmin,
                       vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_aspect("equal")
        ax.set_title(f"t = {t}")
        ax.axis("off")

        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        images.append(image)

        plt.close(fig)

    imageio.mimsave(filename, images, fps=5)


def main():
    key = jr.PRNGKey(0)

    coords, shape = make_grid(nx=15, ny=15, nt=20)

    kernels = {
        "separable": lambda x, y: st_separable_rbf_matern_1_2(
            x, y, 1.0, 0.2, 0.2
        ),
        "nonsep": lambda x, y: st_nonseparable_gneiting(
            x, y, 1.0, 0.2, 0.5, 0.7, 0.5
        ),
        "advected": lambda x, y: st_nonseparable_gneiting_advected(
            x, y, 1.0, 0.2, 0.5, 0.7, 0.5,
            v=jnp.array([0.5, 0.2])
        ),
    }

    all_frames = []

    for i, (name, kernel) in enumerate(kernels.items()):
        subkey = jr.fold_in(key, i)
        f = sample_gp(kernel, coords, subkey)
        frames = reshape_to_frames(f, shape)
        all_frames.append((name, frames))

    # compute global scale
    vmin = float(min(np.min(np.array(fr)) for _, fr in all_frames))
    vmax = float(max(np.max(np.array(fr)) for _, fr in all_frames))

    for name, frames in all_frames:
        save_gif(frames, f"{name}.gif", vmin=vmin, vmax=vmax)


if __name__ == "__main__":
    main()