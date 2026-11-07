"""Unit 1 check: what reduce_scatter / all_gather actually return in mini-nccl.

Input: W=4, length 8, rank r element j = 10*j + r.
Reference: computed independently in the parent process with torch.
"""

import torch

from mini_nccl import collectives as c
from mini_nccl.launcher import run

W, N = 4, 8


def make_input(rank: int) -> torch.Tensor:
    return torch.arange(N, dtype=torch.int64) * 10 + rank


def worker(pg):
    x = make_input(pg.rank)
    before = x.clone()
    rs = c.reduce_scatter(pg, x)
    gathered = c.all_gather(pg, rs)
    return {
        "rs": rs.tolist(),
        "rs_shape": tuple(rs.shape),
        "input_unchanged": torch.equal(x, before),
        "gathered": [g.tolist() for g in gathered],
    }


if __name__ == "__main__":
    ref = sum(make_input(r) for r in range(W))
    print("reference full sum:", ref.tolist())
    for r, out in enumerate(run(worker, W)):
        print(f"rank {r}: reduce_scatter -> {out['rs']} shape={out['rs_shape']} "
              f"input_unchanged={out['input_unchanged']}")
        if r == 2:
            print(f"rank 2: all_gather -> {out['gathered']}")
        assert out["rs"] == ref.view(W, -1)[r].tolist()
    print("all ranks match reference")
