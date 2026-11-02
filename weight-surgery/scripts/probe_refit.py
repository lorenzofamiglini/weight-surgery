"""Is the head the bottleneck? Refit only the head on half of the groups, pulled towards the old head
by an L2 term, score on the other half, swap. Prints per-class precision, recall, F1 and the top
confusions for the old and the refit head. With --only, only the rows of those classes move: the
others stay frozen, so the edit cannot damage them. With --write, fits on every group and writes the
new head.

FEATURES.pt holds {"hidden": float[N, d], "gold": long[N], "groups": list[str]}: the vectors that feed
the head, the gold tag id of each, and its source (file, user, document).

Run: python probe_refit.py FEATURES.pt MODEL_DIR [--pull 1e-2] [--only A,B] [--write OUT_DIR]
"""

import argparse
from collections import Counter

import torch
import torch.nn.functional as F

from weights import class_of, tag_names, tensor, write


def fit(hidden, gold, W0, b0, pull, steps, frozen):
    W, b = W0.clone().requires_grad_(), b0.clone().requires_grad_()
    optimiser = torch.optim.Adam([W, b], lr=1e-3)
    for _ in range(steps):
        optimiser.zero_grad()
        loss = F.cross_entropy(hidden @ W.T + b, gold) + pull * ((W - W0) ** 2).sum()
        loss.backward()
        W.grad[frozen], b.grad[frozen] = 0, 0
        optimiser.step()
    return W.detach(), b.detach()


def report(name, gold, predicted, classes):
    print(f"\n== {name} head ==\n{'class':26}{'gold':>7}{'P':>7}{'R':>7}{'F1':>7}   top confusions")
    for label in classes:
        right = sum(g == label and p == label for g, p in zip(gold, predicted))
        precision, recall = right / max(predicted.count(label), 1), right / max(gold.count(label), 1)
        confused = Counter(p for g, p in zip(gold, predicted) if g == label and p != label).most_common(3)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        print(f"{label:26}{gold.count(label):7}{precision:7.3f}{recall:7.3f}{f1:7.3f}   {confused}")


def main() -> None:
    arguments = argparse.ArgumentParser()
    arguments.add_argument("features")
    arguments.add_argument("model")
    arguments.add_argument("--weight", default="classifier.weight")
    arguments.add_argument("--pull", type=float, default=1e-2)
    arguments.add_argument("--only")
    arguments.add_argument("--steps", type=int, default=200)
    arguments.add_argument("--write")
    options = arguments.parse_args()
    torch.manual_seed(0)
    data = torch.load(options.features)
    hidden, gold, groups = data["hidden"].float(), data["gold"], data["groups"]
    bias_name = options.weight.rsplit(".", 1)[0] + ".bias"
    W0, b0 = tensor(options.model, options.weight).float(), tensor(options.model, bias_name).float()
    tags = [class_of(t) for t in tag_names(options.model, len(W0))]
    moving = set(options.only.split(",")) if options.only else set(tags)
    frozen = torch.tensor([tag not in moving for tag in tags])
    names = sorted(set(groups))
    fold = torch.tensor([names.index(g) % 2 for g in groups])
    old, new, change = torch.empty_like(gold), torch.empty_like(gold), []
    for k in (0, 1):
        W, b = fit(hidden[fold != k], gold[fold != k], W0, b0, options.pull, options.steps, frozen)
        change.append(((W - W0).norm() / W0.norm()).item())
        old[fold == k] = (hidden[fold == k] @ W0.T + b0).argmax(1)
        new[fold == k] = (hidden[fold == k] @ W.T + b).argmax(1)
    truth = [tags[i] for i in gold.tolist()]
    print(f"{len(gold)} vectors, {len(names)} groups, pull {options.pull}, head change {sum(change) / 2:.3%}")
    classes = sorted(set(truth))
    report("old", truth, [tags[i] for i in old.tolist()], classes)
    report("refit (held-out halves)", truth, [tags[i] for i in new.tolist()], classes)
    if options.write:
        W, b = fit(hidden, gold, W0, b0, options.pull, options.steps, frozen)
        write(options.model, options.write, {options.weight: W.to(tensor(options.model, options.weight).dtype),
                                             bias_name: b.to(tensor(options.model, bias_name).dtype)})
        print(f"\nwrote {options.write}: head refit on every group")


if __name__ == "__main__":
    main()
