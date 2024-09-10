#!/usr/bin/env python3
"""
This file analyzes attention stored Flax module intermediates variables.
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
from jax import random
from mpl_toolkits.axes_grid1 import ImageGrid
from sps.gp import GP
from sps.kernels import rbf
from sps.priors import Prior

import dsp.meta_regression.train_utils as tu

# TODO(danj):
# run model over dataset, storing intermediates
# visualize all query tensors at each layer (sort by s_test)
# plot sample query location with lines to context points, with width determined by strength

# Final plots:
# Attention by layer for 6x1 and 1x6
# Tensors by layer for 6x1 and 1x6
# Plot of sample with sample query pointing to context points


def main(args):
    rng = random.key(args.seed)
    rng_gp, rng_extra = random.split(rng)
    s_min, s_max, num_ctx, num_test, ls = -2, 2, 16, 128, 0.2
    s_ctx = jnp.linspace(s_min, s_max, num_ctx)[:, None]  # [L_ctx, 1]
    s_test = jnp.linspace(s_min, s_max, num_test)[:, None]  # [L_test, 1]
    gp = GP(rbf, ls=Prior("fixed", {"value": ls}))
    f_test, *_ = gp.simulate(rng_gp, s_test, batch_size=1)
    f_test = f_test[0]  # [B, L_test, 1] -> [L_test, 1]
    f_ctx = f_test[:: num_test // num_ctx, :]  # [L_ctx, 1]
    state, _ = tu.load_ckpt(Path(args.ckpt_path))
    (f_mu, f_std, *_), vars = state.apply_fn(
        {"params": state.params, **state.kwargs},
        s_ctx[None, ...],  # add batch dim
        f_ctx[None, ...],
        s_test[None, ...],
        mutable="intermediates",
        rngs={"extra": rng_extra},
    )
    fig = tu.plot_posterior_predictive(
        s_ctx.squeeze(),  # squeeze out first and last dims in [B=1, L, D_S=1]
        f_ctx.squeeze(),
        s_test.squeeze(),
        f_test.squeeze(),
        f_mu.squeeze(),
        f_std.squeeze(),
    )
    fig.suptitle(f"RBF GP sample with (var: 1.0, ls: {ls}), seed {args.seed}")
    save(fig, "sample.png")
    path_leaf_tpls = jtu.tree_leaves_with_path(vars["intermediates"])
    cross_attns = [x for path, x in path_leaf_tpls if "cross_attn" in jtu.keystr(path)]
    self_attns = [x for path, x in path_leaf_tpls if "self_attn" in jtu.keystr(path)]
    fig = plot_attn(cross_attns)
    save(fig, "cross_attn.png")
    fig = plot_attn(self_attns)
    save(fig, "self_attn.png")


def save(fig, name: str):
    fig.savefig(name, dpi=150)
    plt.clf()
    plt.close(fig)


def plot_attn(attns: list[jax.Array]):
    num_layers, num_heads = len(attns), attns[0].shape[1]  # [B, H, L_test, L_ctx]
    fig = plt.figure(figsize=(num_heads * 2, num_layers * 2))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(num_layers, num_heads),
        axes_pad=0.15,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="7%",
        cbar_pad=0.15,
    )
    for i in range(num_layers):
        for h in range(num_heads):
            k = i * num_heads + h
            attn_head = attns[i][0, h]
            im = grid[k].imshow(attn_head, cmap="inferno")
            grid[k].set_ylabel(f"Layer {i+1}")
            grid[k].set_xlabel(f"Head {h+1}")
    grid[0].cax.colorbar(im)
    plt.tight_layout()
    return fig


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "ckpt_path",
        help="Path to a 1D GP model checkpoint.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=7,
        help="Root seed for all random operations.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args)
