"""Which neurons vote for class A against class B? Weights only, no forward pass.

The head reads the direction d = mean of A's tag rows - mean of B's rows (times the final norm's gain,
with --norm). Every matrix that writes into the residual stream has one column per neuron (or channel);
d^T W gives each column's vote: positive pushes towards A, negative towards B. Votes are shown as the
cosine of the column with d, in units of a random direction's (1/sqrt(d)): a column at 1 is noise, a
column above about 4 is aligned beyond chance. With --embedding, the same for every vocabulary token. Exact for pre-norm residual models up to the final norm's
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
                W = W.float()
                votes = (d @ W) / W.norm(dim=0).clamp_min(1e-12) * len(d) ** 0.5
                found.append((votes.abs().max().item(), name, votes))
    print(f"direction {a} - {b}, {len(found)} writing matrices; column cosine with it in units of a random direction\n")
    print(f"{'tensor':56}{'max':>6}   top columns towards {a} | towards {b}")
    for peak, name, votes in sorted(found, reverse=True, key=lambda f: f[0]):
        up, down = votes.topk(options.top // 2), (-votes).topk(options.top // 2)
        cells = " ".join(f"{i}:{v:+.1f}" for v, i in zip(up.values.tolist(), up.indices.tolist()))
        cells += " | " + " ".join(f"{i}:{-v:+.1f}" for v, i in zip(down.values.tolist(), down.indices.tolist()))
        print(f"{name:56}{peak:6.1f}   {cells}")
    if options.embedding:
        E = tensor(options.model, options.embedding).float()
        scores = E @ d / E.norm(dim=1).clamp_min(1e-12) * len(d) ** 0.5
        words = vocabulary(options.model)
        for label, picked in ((a, scores.topk(options.top * 2)), (b, (-scores).topk(options.top * 2))):
            print(f"\ntokens whose embedding points towards {label}: "
                  + ", ".join(f"{words.get(i, i)!r} {abs(v):.1f}" for v, i in zip(picked.values.tolist(), picked.indices.tolist())))


if __name__ == "__main__":
    main()
