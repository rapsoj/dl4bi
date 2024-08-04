#!/usr/bin/env python3
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    d = recursive_defaultdict()
    for path in Path("results/gp").rglob("*.npy"):
        d_tmp = d
        for dir in path.parts[:-2]:
            d_tmp = d_tmp[dir]
        model = path.stem
        if not isinstance(d_tmp[model], np.ndarray):
            d_tmp[model] = np.array([])
        # stack regrets by seed for each model
        d_tmp[model] = np.hstack([d_tmp[model], np.load(path)[:, -1]])
    with open("bayes_opt_summary.pkl", "wb") as f:
        pickle.dump(summarize(d), f)


def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)


def summarize(d):
    if isinstance(d, defaultdict):
        d = {k: summarize(v) for k, v in d.items()}
    elif isinstance(d, np.ndarray):
        return {"mean": d.mean(), "std": d.std()}
    return d


if __name__ == "__main__":
    main()
