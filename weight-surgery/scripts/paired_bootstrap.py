"""Paired difference between two systems on the same examples, with a 95% interval from a bootstrap
over source groups. Each file has one JSON line per example, in the same order, with "group" and
either "score" (the metric is the mean) or "tp", "fp", "fn" (the metric is micro-F1).

Run: python paired_bootstrap.py BASE.jsonl EDIT.jsonl [--resamples 1000]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def metric(rows: list[dict]) -> float:
    if "score" in rows[0]:
        return sum(r["score"] for r in rows) / len(rows)
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def main() -> None:
    arguments = argparse.ArgumentParser()
    arguments.add_argument("base")
    arguments.add_argument("edit")
    arguments.add_argument("--resamples", type=int, default=1000)
    options = arguments.parse_args()
    base, edit = ([json.loads(line) for line in Path(p).read_text().splitlines() if line] for p in (options.base, options.edit))
    if len(base) != len(edit):
        raise ValueError("the files must hold the same examples in the same order")
    groups = defaultdict(list)
    for index, row in enumerate(base):
        groups[row["group"]].append(index)
    keys, rng, differences = list(groups), random.Random(0), []
    for _ in range(options.resamples):
        picks = [i for _ in keys for i in groups[keys[rng.randrange(len(keys))]]]
        differences.append(metric([edit[i] for i in picks]) - metric([base[i] for i in picks]))
    differences.sort()
    low, high = differences[int(0.025 * options.resamples)], differences[int(0.975 * options.resamples) - 1]
    print(f"base {metric(base):.4f}  edit {metric(edit):.4f}  difference {metric(edit) - metric(base):+.4f} "
          f"[{low:+.4f}, {high:+.4f}]  ({len(keys)} groups)")


if __name__ == "__main__":
    main()
