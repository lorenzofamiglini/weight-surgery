"""Which neurons vote for class A against class B? Weights only, no forward pass.

The head reads the direction d = mean of A's tag rows - mean of B's rows (times the final norm's gain,
with --norm). Every matrix that writes into the residual stream has one column per neuron (or channel);
d^T W gives each column's vote: positive pushes towards A, negative towards B. With --embedding, the
same for every token of the vocabulary. Exact for pre-norm residual models up to the final norm's
per-token scale; for post-norm models (BERT) the later norms bend it, so treat it as a ranking.

Run: python weight_votes.py MODEL_DIR --pair A,B [--norm NAME] [--writers REGEX] [--embedding NAME]
       [--weight classifier.weight] [--top 10] [--out DIRECTION.pt]
"""

import argparse
import json
import re
from pathlib import Path

import torch

from weights import class_of, shards, tag_names, tensor

WRITERS = r"(down_proj|o_proj|out_proj|fc2|output\.dense|w2|c_proj)\.weight$"


def direction(directory: str, weight: str, a: str, b: str) -> torch.Tensor:
    W = tensor(directory, weight).float()
    tags = [class_of(t) for t in tag_names(directory, len(W))]
    rows = {name: W[[i for i, t in enumerate(tags) if t == name]] for name in (a, b)}
    for name, values in rows.items():
        if not len(values):
            raise SystemExit(f"no head rows for {name}; classes: {sorted(set(tags))}")
    return rows[a].mean(0) - rows[b].mean(0)


def vocabulary(directory: str) -> dict[int, str]:
    path = Path(directory) / "tokenizer.json"
    if not path.exists():
        lines = Path(directory) / "vocab.txt"
        return dict(enumerate(lines.read_text().splitlines())) if lines.exists() else {}
    vocab = json.loads(path.read_text())["model"]["vocab"]
    return {i: entry[0] for i, entry in enumerate(vocab)} if isinstance(vocab, list) else {i: t for t, i in vocab.items()}


def main() -> None:
    arguments = argparse.ArgumentParser()
    arguments.add_argument("model")
    arguments.add_argument("--pair", required=True)
    arguments.add_argument("--weight", default="classifier.weight")
    arguments.add_argument("--norm")
    arguments.add_argument("--writers", default=WRITERS)
    arguments.add_argument("--embedding")
    arguments.add_argument("--top", type=int, default=10)
    arguments.add_argument("--out")
    options = arguments.parse_args()
    a, b = options.pair.split(",")
    d = direction(options.model, options.weight, a, b)
    if options.norm:
        d = d * tensor(options.model, options.norm).float()
    d = d / d.norm()
    if options.out:
        torch.save(d, options.out)
    found = []
    for values in shards(options.model).values():
        for name, W in values.items():
            if re.search(options.writers, name) and W.dim() == 2 and W.shape[0] == len(d):
                votes = d @ W.float()
                found.append((votes.norm().item() / W.float().norm().item(), name, votes))
    print(f"direction {a} - {b}, {len(found)} writing matrices, sorted by how much of each points along it\n")
    print(f"{'tensor':56}{'share':>8}   top columns towards {a} | towards {b}")
    for share, name, votes in sorted(found, reverse=True, key=lambda f: f[0]):
        up, down = votes.topk(options.top // 2), (-votes).topk(options.top // 2)
        cells = " ".join(f"{i}:{v:+.2f}" for v, i in zip(up.values.tolist(), up.indices.tolist()))
        cells += " | " + " ".join(f"{i}:{-v:+.2f}" for v, i in zip(down.values.tolist(), down.indices.tolist()))
        print(f"{name:56}{share:8.4f}   {cells}")
    if options.embedding:
        scores = tensor(options.model, options.embedding).float() @ d
        words = vocabulary(options.model)
        for label, picked in ((a, scores.topk(options.top * 2)), (b, (-scores).topk(options.top * 2))):
            print(f"\ntokens whose embedding points towards {label}: "
                  + ", ".join(repr(words.get(i, i)) for i in picked.indices.tolist()))


if __name__ == "__main__":
    main()
