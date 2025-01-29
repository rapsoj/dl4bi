import json
from collections import defaultdict
from pathlib import Path
from time import time

import hydra
from jax import random
from omegaconf import DictConfig
from tqdm import tqdm

from dl4bi.meta_regression.train_utils import load_ckpt

from .sir import build_dataloader, instantiate, select_steps


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
        state, _ = load_ckpt(path)
        batch = next(batches)
        valid_step(rng, state, batch)  # pre-compile
        pbar = tqdm(
            range(cfg.valid_num_steps),
            unit=" samples",
            leave=False,
            dynamic_cols=True,
        )
        metrics = defaultdict(list)
        for i in pbar:
            rng_i, rng = random.split(rng)
            batch = next(batches)
            start = time()
            try:
                m = valid_step(rng_i, state, batch)
            except Exception:
                print("OOM!")
            end = time()
            m["s_elapsed"] = end - start
            for k, v in m.items():
                metrics[k] += [v]
        name = path.stem + "_" + cfg.data.name
        out_path = (path.parent / name).with_suffix(".json")
        with open(out_path, "w") as f:
            json.dump(defaultdict, f, indent=2)


if __name__ == "__main__":
    main()
