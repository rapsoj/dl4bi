#!/usr/bin/env python3
import json
from pathlib import Path
from time import time

import hydra
import jax.numpy as jnp
from jax import jit, random
from jaxlib.xla_client import XlaRuntimeError
from omegaconf import DictConfig

from dl4bi.meta_regression.train_utils import (
    build_gp_dataloader,
    cfg_to_run_name,
    load_ckpt,
)


@hydra.main("configs/gp", config_name="default", version_base=None)
def main(cfg: DictConfig):
    num_exp = 6  # TODO(danj): change back to 6
    num_trials = 100
    cfg.data.batch_size = 1
    cfg.data.num_ctx.min = 100
    cfg.data.num_ctx.max = 100
    rng = random.key(cfg.seed)
    s_dim = len(cfg.data.s)
    min_s = jnp.array([axis["start"] for axis in cfg.data.s])
    max_s = jnp.array([axis["stop"] for axis in cfg.data.s])
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    path = f"results/gp/{cfg.data.name}/{cfg.kernel.kwargs.kernel.func}/{cfg.seed}/{run_name}"
    path = Path(path)
    state, _ = load_ckpt(path.with_suffix(".ckpt"))
    dataloader = build_gp_dataloader(cfg.data, cfg.kernel)
    s_ctx, f_ctx, valid_lens_ctx, *_ = next(dataloader(rng))

    @jit
    def apply(s_test):
        return state.apply_fn(
            {"params": state.params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            valid_lens_test=None,
            rngs={"extra": rng},
        )

    results = {}
    for exp in range(num_exp + 1):  # 10^0 -> 10^num_exp
        num_points = 10**exp
        s_test = random.uniform(
            rng,
            (cfg.data.num_ctx.max, s_dim),
            minval=min_s,
            maxval=max_s,
        )[None, ...]  # add batch dim
        # run once to JIT and if it OOMs, skip this and remaining tests
        try:
            apply(s_test)
        except XlaRuntimeError:  # OOM
            print(results)
            with open(path.with_suffix("_runtimes.json"), "w") as f:
                json.dump(results, f, indent=2, sort_keys=True)
            return
        start = time()
        for i in range(num_trials):
            apply(s_test)
        stop = time()
        results[num_points] = (stop - start) / num_trials
    print(results)
    with open(path.with_suffix("_runtimes.json"), "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
