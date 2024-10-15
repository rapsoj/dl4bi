#!/usr/bin/env python3
import argparse
import sys
from glob import glob

from omegaconf import DictConfig, OmegaConf

from dl4bi.meta_regression.train_utils import (
    build_gp_dataloader,
    load_ckpt,
    log_posterior_predictive_plots,
)


def main(args):
    for ckpt in glob(args.ckpt_dir + "/**.ckpt"):
        state, _ = load_ckpt(ckpt)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "ckpt_dir",
        help="Path to a 2D directory of checkpoints.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=7,
        type=int,
        help="Root seed for all random operations.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args)
