#!/usr/bin/env python3
import argparse
import sys

import pandas as pd

# NOTE: first download the results you care about from wandb,
# e.g. name, kernel, seed, runtime, valid_loss, test_loss, etc
# then run this script to generate latex tables summarizing it


def main(args):
    df = pd.read_csv(args.path)
    if "loss" in args.col:
        df["log_likelihood"] = -df[args.col]
        args.col = "log_likelihood"
    funcs = ["mean", "std"]
    x = df[[*args.group_by, args.col]].groupby(args.group_by).agg(funcs)
    x.columns = funcs
    x = x.apply(lambda r: f"${r['mean']:.2f}\\pm{r['std']:.2f}$", axis=1)
    x = x.reset_index()
    colnames = list(x.columns)
    colnames[-1] = args.col
    x.columns = colnames
    if args.pivot:
        rows = list(set(x.columns) - set([args.pivot, args.col]))
        x = x.pivot(index=rows, columns=args.pivot, values=args.col)
        x = x.reset_index()
    x.columns = [c.title().replace("_", " ") for c in x.columns]
    x_tex = x.to_latex(
        index=False,
        caption=args.name,
        label=args.name.lower().replace(" ", "-"),
        float_format="{:0.2f}".format,
        column_format="l" + "r" * (len(x.columns) - 1),
    )
    print(x_tex)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", help="Path to csv.")
    parser.add_argument(
        "-c",
        "--col",
        default="test_loss",
        help="Column to apply funcs.",
    )
    parser.add_argument(
        "-g",
        "--group_by",
        nargs="+",
        default=["Name"],
        help="Columns to group by.",
    )
    parser.add_argument(
        "-p",
        "--pivot",
        nargs="?",
        default=None,
        help="Pivot to column headers.",
    )
    parser.add_argument(
        "-n",
        "--name",
        default="My Experiment",
        help="Name to use for caption and label.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args)
