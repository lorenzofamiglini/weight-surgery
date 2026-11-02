"""Remove a direction v from what a matrix writes (len(v) = rows: W <- W - s v (v^T W)) or reads
(len(v) = columns: W <- W - s (W v) v^T), only on the K columns (or rows) that carry most of it.
Writes a new model directory and prints how many weights changed.

Run: python ablate_direction.py MODEL_DIR DIRECTION.pt --tensor NAME [--tensor NAME]
       --out OUT_DIR [--scale 1.0] [--columns K]
"""

import argparse

import torch

from weights import shards, write


def ablated(W: torch.Tensor, v: torch.Tensor, scale: float, columns: int | None) -> tuple[torch.Tensor, int]:
    v = v / v.norm()
    if len(v) == W.shape[0]:
        delta = scale * torch.outer(v, v @ W)
        weight = (v @ W).abs()
        keep = torch.zeros(W.shape[1], dtype=torch.bool)
        keep[weight.topk(columns or W.shape[1]).indices] = True
        delta[:, ~keep] = 0
    elif len(v) == W.shape[1]:
        delta = scale * torch.outer(W @ v, v)
        weight = (W @ v).abs()
        keep = torch.zeros(W.shape[0], dtype=torch.bool)
        keep[weight.topk(columns or W.shape[0]).indices] = True
        delta[~keep] = 0
    else:
        raise ValueError(f"direction of size {len(v)} fits neither side of {tuple(W.shape)}")
    return W - delta, int((delta != 0).sum())


def main() -> None:
    arguments = argparse.ArgumentParser()
    arguments.add_argument("model")
    arguments.add_argument("direction")
    arguments.add_argument("--tensor", action="append", required=True)
    arguments.add_argument("--out", required=True)
    arguments.add_argument("--scale", type=float, default=1.0)
    arguments.add_argument("--columns", type=int)
    options = arguments.parse_args()
    v = torch.load(options.direction).float()
    everything = {name: value for values in shards(options.model).values() for name, value in values.items()}
    total = sum(value.numel() for value in everything.values())
    changed, count = {}, 0
    for name in options.tensor:
        W, edited = ablated(everything[name].float(), v, options.scale, options.columns)
        changed[name], count = W.to(everything[name].dtype), count + edited
    write(options.model, options.out, changed)
    print(f"wrote {options.out}: {count} weights changed, {count / total:.4%} of {total}")


if __name__ == "__main__":
    main()
