import json
from collections import defaultdict
from pathlib import Path
from time import time

import hydra
from jax import random
from omegaconf import DictConfig
from sir import build_dataloader, instantiate, select_steps
from tqdm import tqdm

from dl4bi.meta_regression.train_utils import load_ckpt


@hydra.main("configs/sir", config_name="default", version_base=None)
def main(cfg: DictConfig):
    results_path = Path(f"results/{cfg.project}/{cfg.seed}")
    rng_data, rng_valid = random.split(random.key(cfg.seed))
    dataloader = build_dataloader(cfg.data, cfg.sim)
    model = instantiate(cfg.model)
    _, valid_step = select_steps(model, is_categorical=True)
    for path in results_path.rglob("*.ckpt"):
        rng = rng_valid
        batches = dataloader(rng_data)
        out_fn = path.parent / (path.stem + "_" + cfg.data.name)
        state, _ = load_ckpt(path)
        batch = next(batches)
        try:
            valid_step(rng, state, batch)  # precompile
        except Exception:
            with open(out_fn.with_suffix(".txt"), "w") as f:
                f.write("OOM\n")
            continue
        pbar = tqdm(
            range(cfg.valid_num_steps),
            unit=" samples",
            leave=False,
            dynamic_ncols=True,
        )
        metrics = defaultdict(list)
        for i in pbar:
            rng_i, rng = random.split(rng)
            batch = next(batches)
            start = time()
            m = valid_step(rng_i, state, batch)
            end = time()
            m["s_elapsed"] = end - start
            for k, v in m.items():
                metrics[k] += [float(v)]
        with open(out_fn.with_suffix(".json"), "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
