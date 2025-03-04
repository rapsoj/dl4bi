import re
from pathlib import Path
from typing import Union

import jax
import jax.numpy as jnp
from jax import jit, random
from omegaconf import DictConfig, OmegaConf

from ..core.train import TrainState, load_ckpt


def cfg_to_run_name(cfg: DictConfig):
    name = cfg.model.cls
    if "TNPKR" in name:
        prefix = "model.kwargs.blk.kwargs.attn."
        attn_cls = OmegaConf.select(cfg, prefix + "cls")
        if attn_cls == "MultiHeadAttention":
            attn_cls = OmegaConf.select(cfg, prefix + "kwargs.attn.cls")
        name += ": " + attn_cls
    return name


def load_ckpts(
    dir: Union[str, Path],
    only_regex: Union[str, re.Pattern] = r".*",
    exclude_regex: Union[str, re.Pattern] = "$^",
):
    """Loads all checkpoints in a given base dir."""
    ckpt = {}
    if isinstance(only_regex, str):
        only_regex = re.compile(only_regex, re.IGNORECASE)
    if isinstance(exclude_regex, str):
        exclude_regex = re.compile(exclude_regex, re.IGNORECASE)
    for p in Path(dir).glob("*.ckpt"):
        if only_regex.match(str(p)) and not exclude_regex.match(str(p)):
            state, tmp_cfg = load_ckpt(p)
            ckpt[cfg_to_run_name(tmp_cfg)] = {"state": state, "cfg": tmp_cfg}
    return ckpt


def sample(
    rng: jax.Array,
    state: TrainState,
    s_ctx: jax.Array,  # [L_ctx, D_S]
    f_ctx: jax.Array,  # [L_ctx, D_F]
    s_test: jax.Array,  # [L_test, D_S]
    batch_size: int = 32,
):
    @jit
    def apply(s_ctx, f_ctx, s_test, valid_lens_ctx, rng_extra):
        return state.apply_fn(
            {"params": state.params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            training=False,
            rngs={"extra": rng_extra},
        )

    B, L_ctx, L_test = batch_size, s_ctx.shape[0], s_test.shape[0]
    s_ctx = jnp.repeat(jnp.vstack([s_ctx, s_test])[None, ...], B, axis=0)
    f_ctx = jnp.repeat(jnp.pad(f_ctx, ((0, L_test), (0, 0)))[None, ...], B, axis=0)
    s_test = jnp.repeat(s_test[None, ...], B, axis=0)
    valid_lens_ctx = jnp.repeat(L_ctx, B)
    for i in range(L_test):
        rng_extra, rng_eps, rng = random.split(rng, 3)
        output = apply(s_ctx, f_ctx, s_test, valid_lens_ctx, rng_extra)
        if isinstance(output[1], tuple):  # latent
            output, _ = output  # throw away latent samples
        f_mu, f_std = output
        f_mu_i, f_std_i = f_mu[:, i, :], f_std[:, i, :]
        f_test_i = f_mu_i + f_std_i * random.normal(rng_eps, f_std_i.shape)
        f_ctx = f_ctx.at[:, L_ctx + i, :].set(f_test_i)
        valid_lens_ctx += 1
    return s_ctx[:, L_ctx:, :], f_ctx[:, L_ctx:, :]  # only return test locations


def regression_to_rgb(f: jax.Array):
    return jnp.clip(f / 2 + 0.5, 0, 1)  # [-1, 1] => [0, 1]
