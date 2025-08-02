#!/usr/bin/env -S PYENV_VERSION=torch python3
import argparse
import sys
import time

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention

torch.set_default_device("cuda:0")

# NOTE: cannot get this to scale past ~L=1600
# issue: https://github.com/pytorch/pytorch/issues/152593


def main(seed: int, N: int, B: int, H: int, L: int, D: int, D_s: int):
    torch.manual_seed(seed)
    bias = RBFBias(num_heads=4, num_s=5, num_t=3)
    kernel_options = {}
    # kernel_options = { # NOTE: doesn't help
    #     "BLOCK_M": 64,
    #     "BLOCK_N": 64,
    #     "BLOCK_M1": 32,
    #     "BLOCK_N1": 64,
    #     "BLOCK_M2": 64,
    #     "BLOCK_N2": 32,
    # }  # https://github.com/pytorch/pytorch/issues/133254
    attn = BiasedFlexAttention(bias, kernel_options)
    # NOTE: doesn't support dynamic=True, max-autotune, or reductions in bias
    attn = torch.compile(attn)
    b = sample_batch(B, H, L, D, D_s)
    attn(**b)  # precompile flex_attention (?)
    torch.cuda.synchronize()
    times_fwd = torch.zeros(N)
    times_bwd = torch.zeros(N)
    for i in range(N):
        b = sample_batch(B, H, L, D, D_s)
        torch.cuda.synchronize()
        start_fwd = time.perf_counter()
        q = attn(**b)
        torch.cuda.synchronize()
        stop_fwd = time.perf_counter()
        torch.cuda.synchronize()
        start_bwd = time.perf_counter()
        q.mean().backward()
        torch.cuda.synchronize()
        stop_bwd = time.perf_counter()
        times_fwd[i] = stop_fwd - start_fwd
        times_bwd[i] = stop_bwd - start_bwd
    print(
        f"[B={B} H={H} L={L} D={D} D_s={D_s}]: {times_fwd.mean():0.6f}±{times_fwd.std():0.6f}s, {times_bwd.mean():0.6f}±{times_bwd.std():0.6f}s"
    )


# NOTE: this seems to have trouble with the backward pass, even without
# reductions; performance for biased flex attention and biased scan attention
# for N=100, and various L:
# L=256 BFA: (0.000252, 0.024034), BSA: (0.000169, 0.000492), BFA / BSA: (1.49, 48.8)
# L=512 BFA: (0.000395, 0.089612), BSA: (0.001139, 0.00391), BFA / BSA: (0.35, 22.92)
# L=1024 BFA: (0.001151, 0.386958), BSA: (0.005063, 0.016702), BFA / BSA: (0.23, 23.2)
# L=2048 BFA: (0.003596, 1.050770), BSA: (0.020340, 0.086891), BFA / BSA: (0.18, 12.1)
# L=4096 BFA: (0.013071, 4.199244), BSA: (0.080129, 0.329538), BFA / BSA: (0.16, 12.7)
class RBFBias(nn.Module):
    def __init__(self, num_heads: int, num_s: int, num_t: int):
        super().__init__()
        # reductions like .sum() and higher order params aren't
        # supported by torch.compile in FlexAttention, so specify
        # each var here
        self.num_s = num_s
        self.num_t = num_t
        for dim, num in [("s", num_s), ("t", num_t)]:
            for i in range(num):
                alpha = nn.Parameter(torch.randn(num_heads))
                beta = nn.Parameter(torch.randn(num_heads))
                self.register_parameter(f"{dim}_alpha_{i}", alpha)
                self.register_parameter(f"{dim}_beta_{i}", beta)

    def forward(self, score, b, h, q_idx, kv_idx, qs_s, ks_s, qs_t, ks_t):
        q_s, k_s = qs_s[b, q_idx], ks_s[b, kv_idx]
        q_t, k_t = qs_t[b, q_idx], ks_t[b, kv_idx]
        d_s_sq = torch.square(q_s - k_s).sum()
        d_t_sq = torch.square(q_t - k_t).sum()
        b_rbf = 0.0
        for i in range(self.num_s):
            alpha = getattr(self, f"s_alpha_{i}")[h]
            beta = getattr(self, f"s_beta_{i}")[h]
            b_rbf += alpha * torch.exp(-beta * d_s_sq)
        for i in range(self.num_t):
            alpha = getattr(self, f"t_alpha_{i}")[h]
            beta = getattr(self, f"t_beta_{i}")[h]
            b_rbf += alpha * torch.exp(-beta * d_t_sq)
        return score + b_rbf


class BiasedFlexAttention(nn.Module):
    def __init__(self, bias: nn.Module, kernel_options: dict = {}):
        super().__init__()
        self.bias = bias
        self.kernel_options = kernel_options

    def forward(self, qs, ks, vs, qs_s, ks_s, qs_t, ks_t):
        def score_mod(score, b, h, q_idx, kv_idx):
            return self.bias(score, b, h, q_idx, kv_idx, qs_s, ks_s, qs_t, ks_t)

        return flex_attention(
            qs,
            ks,
            vs,
            score_mod=score_mod,
            kernel_options=self.kernel_options,
        )


def sample_batch(B: int, H: int, L: int, D: int, D_s: int):
    qs, ks, vs = torch.randn(3, B, H, L, D)
    qs_s, ks_s = torch.randn(2, B, L, D_s)
    qs_t, ks_t = torch.randn(2, B, L, 1)
    return {
        "qs": qs,
        "ks": ks,
        "vs": vs,
        "qs_s": qs_s,
        "ks_s": ks_s,
        "qs_t": qs_t,
        "ks_t": ks_t,
    }


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-seed",
        type=int,
        default=42,
        help="Seed.",
    )
    parser.add_argument(
        "-N",
        type=int,
        default=100,
        help="Number of trials.",
    )
    parser.add_argument(
        "-B",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "-L",
        type=int,
        default=1664,  # 1024 + 512 + 128
        help="Length of sequence to test.",
    )
    parser.add_argument(
        "-H",
        type=int,
        default=4,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "-D",
        type=int,
        default=64,
        help="Embedding dim per head.",
    )
    parser.add_argument(
        "-D_s",
        type=int,
        default=2,
        help="Spatial dimension.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(**vars(args))
