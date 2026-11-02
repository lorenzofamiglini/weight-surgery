"""Which classes of a classification head use nearly the same direction?

One direction per class: the mean of its tag rows (B-/I-/E-/S- grouped) minus the null row. The mean
of all class directions (the shared "is a span" part) is removed, then every pair is compared by cosine.
The top pairs of the first model are printed for every model given, so you see whether training moved
them apart. Also prints per-class norm and bias and the head's singular values.

Run: python head_geometry.py MODEL_DIR [OTHER_DIR ...] [--weight classifier.weight] [--null O]
"""

import argparse

import torch
import torch.nn.functional as F

from weights import class_of, tag_names, tensor


def directions(directory: str, weight: str, null: str) -> tuple[dict[str, torch.Tensor], dict]:
    W = tensor(directory, weight).float()
    bias_name = weight.rsplit(".", 1)[0] + ".bias"
    try:
        b = tensor(directory, bias_name).float()
    except KeyError:
        b = torch.zeros(len(W))
    tags = tag_names(directory, len(W))
    base = W[tags.index(null)] if null in tags else torch.zeros(W.shape[1])
    groups = {}
    for index, tag in enumerate(tags):
        if tag != null:
            groups.setdefault(class_of(tag), []).append(index)
    found = {name: W[ids].mean(0) - base for name, ids in groups.items()}
    common = torch.stack(list(found.values())).mean(0)
    stats = {name: (W[ids].norm(dim=1).mean().item(), b[ids].mean().item()) for name, ids in groups.items()}
    stats["_singular"] = torch.linalg.svdvals(W - W.mean(0))[:8].tolist()
    return {name: d - common for name, d in found.items()}, stats


def main() -> None:
    arguments = argparse.ArgumentParser()
    arguments.add_argument("models", nargs="+")
    arguments.add_argument("--weight", default="classifier.weight")
    arguments.add_argument("--null", default="O")
    arguments.add_argument("--top", type=int, default=15)
    options = arguments.parse_args()
    loaded = {m: directions(m, options.weight, options.null) for m in options.models}
    first, stats = loaded[options.models[0]]
    names = sorted(first)
    pairs = sorted(((F.cosine_similarity(first[a], first[b], dim=0).item(), a, b)
                    for i, a in enumerate(names) for b in names[i + 1:]), reverse=True)[: options.top]
    print(f"{'pair':44}" + "".join(f"{m.rstrip('/').split('/')[-1][:14]:>15}" for m in options.models))
    for _, a, b in pairs:
        cells = []
        for found, _ in loaded.values():
            cells.append(f"{F.cosine_similarity(found[a], found[b], dim=0).item():+15.3f}" if a in found and b in found else f"{'-':>15}")
        print(f"{a + ' ~ ' + b:44}" + "".join(cells))
    print(f"\n{'class':28}{'norm':>8}{'bias':>9}")
    for name in sorted(names, key=lambda n: stats[n][1]):
        print(f"{name:28}{stats[name][0]:8.3f}{stats[name][1]:+9.3f}")
    print("\nsingular values:", " ".join(f"{s:.2f}" for s in stats["_singular"]))


if __name__ == "__main__":
    main()
