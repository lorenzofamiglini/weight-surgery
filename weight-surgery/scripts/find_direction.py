"""Is a behaviour written along one direction? The difference of the class means of passing (flag 0)
and failing (flag 1) vectors, fitted on half of the groups, scored by AUC of the projection on the
other half, then swapped. Run it on each layer's vectors: the layer where AUC jumps writes the
behaviour. With --out, saves the unit direction fitted on every group.

FEATURES.pt holds {"hidden": float[N, d], "flag": long[N], "groups": list[str]}.

Run: python find_direction.py FEATURES.pt [--out DIRECTION.pt]
"""

import argparse

import torch


def auc(scores: torch.Tensor, flags: torch.Tensor) -> float:
    ranks = scores.argsort().argsort().float() + 1
    positive = flags.sum().item()
    negative = len(flags) - positive
    return ((ranks[flags == 1].sum() - positive * (positive + 1) / 2) / max(positive * negative, 1)).item()


def direction(hidden: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    found = hidden[flags == 1].mean(0) - hidden[flags == 0].mean(0)
    return found / found.norm()


def main() -> None:
    arguments = argparse.ArgumentParser()
    arguments.add_argument("features")
    arguments.add_argument("--out")
    options = arguments.parse_args()
    data = torch.load(options.features)
    hidden, flags, groups = data["hidden"].float(), data["flag"], data["groups"]
    names = sorted(set(groups))
    fold = torch.tensor([names.index(g) % 2 for g in groups])
    if any(len(set(flags[fold == k].tolist())) < 2 for k in (0, 1)):
        raise ValueError("each half of the groups needs passing and failing vectors")
    scores = [auc(hidden[fold == k] @ direction(hidden[fold != k], flags[fold != k]), flags[fold == k]) for k in (0, 1)]
    print(f"{len(flags)} vectors ({int(flags.sum())} failing), {len(names)} groups")
    print(f"held-out AUC {scores[0]:.3f} / {scores[1]:.3f}  (0.5 = no direction, 1.0 = one clean direction)")
    if options.out:
        torch.save(direction(hidden, flags), options.out)
        print(f"saved {options.out}")


if __name__ == "__main__":
    main()
