"""What changed between two checkpoints of the same network? Per tensor: relative change
|B - A| / |A|, and for matrices the share of the change held by its top singular direction (near 1:
one targeted change; small: spread out). Tensors present in only one checkpoint, or with a new shape,
are listed apart. Prints the most changed tensors and the change per layer.

Run: python diff_checkpoints.py A_DIR B_DIR [--top 20]
"""

import argparse
import re
from collections import defaultdict

import torch

from weights import shards


def flat(directory: str) -> dict[str, torch.Tensor]:
    return {name: value for values in shards(directory).values() for name, value in values.items()}


def main() -> None:
    arguments = argparse.ArgumentParser()
    arguments.add_argument("a")
    arguments.add_argument("b")
    arguments.add_argument("--top", type=int, default=20)
    options = arguments.parse_args()
    a, b = flat(options.a), flat(options.b)
    rows, layers, apart = [], defaultdict(lambda: [0.0, 0.0]), []
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b or a[name].shape != b[name].shape:
            apart.append(f"{name}: {tuple(a[name].shape) if name in a else '-'} -> {tuple(b[name].shape) if name in b else '-'}")
            continue
        old, delta = a[name].float(), b[name].float() - a[name].float()
        change = (delta.norm() / old.norm().clamp_min(1e-12)).item()
        top = None
        if delta.dim() == 2 and delta.norm() > 0:
            singular = torch.linalg.svdvals(delta)
            top = (singular[0] ** 2 / (singular ** 2).sum()).item()
        rows.append((change, name, top))
        layer = re.search(r"layers?\.(\d+)\.", name)
        key = f"layer {int(layer[1]):3}" if layer else "other"
        layers[key][0] += (delta ** 2).sum().item()
        layers[key][1] += (old ** 2).sum().item()
    print(f"{'relative change':>16}  {'top-1 share':>11}  tensor")
    for change, name, top in sorted(rows, reverse=True)[: options.top]:
        print(f"{change:16.4f}  {'' if top is None else f'{top:11.3f}':>11}  {name}")
    print("\nchange per layer:")
    for key in sorted(layers):
        moved, size = layers[key]
        print(f"  {key}  {(moved / max(size, 1e-12)) ** 0.5:.4f}")
    if apart:
        print("\nonly in one checkpoint, or reshaped:\n  " + "\n  ".join(apart))


if __name__ == "__main__":
    main()
