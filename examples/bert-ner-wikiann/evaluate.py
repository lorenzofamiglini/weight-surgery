"""Step 6: entity-level scores of a model directory on a WikiANN split. Writes one JSON line per
sentence (group, tp, fp, fn, and the same per class) for paired_bootstrap.py, and prints micro-F1 and
per-class F1. --drop CLASS adds -1e4 to that class's biases (the cheap baseline).

Run: python evaluate.py MODEL_DIR test out.jsonl [--drop MISC]
"""

import argparse
import json
from collections import Counter

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import split


def spans(tags: list[str]) -> set[tuple[int, int, str]]:
    found, start, kind = set(), None, None
    for i, tag in enumerate(tags + ["O"]):
        prefix, _, label = tag.partition("-")
        if kind and (prefix != "I" or label != kind):
            found.add((start, i, kind)); kind = None
        if prefix == "B" or (prefix == "I" and label != kind):
            start, kind = i, label
    return found


def main() -> None:
    arguments = argparse.ArgumentParser()
    arguments.add_argument("model")
    arguments.add_argument("split")
    arguments.add_argument("out")
    arguments.add_argument("--drop")
    options = arguments.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(options.model)
    network = AutoModelForTokenClassification.from_pretrained(options.model).eval()
    names = network.config.id2label
    if options.drop:
        with torch.no_grad():
            for i, tag in names.items():
                if tag.endswith("-" + options.drop):
                    network.classifier.bias[i] -= 1e4
    rows, lines, total = split(options.split), [], Counter()
    for start in range(0, len(rows), 64):
        batch = rows[start:start + 64]
        encoded = tokenizer([r["words"] for r in batch], is_split_into_words=True, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            predicted = network(**encoded).logits.argmax(-1)
        for b, row in enumerate(batch):
            tags, seen = ["O"] * len(row["words"]), set()
            for position, word in enumerate(encoded.word_ids(b)):
                if word is not None and word not in seen:
                    seen.add(word); tags[word] = names[predicted[b, position].item()]
            guess, truth = spans(tags), spans(row["tags"])
            counts = Counter()
            for s in guess:
                counts["tp" if s in truth else "fp"] += 1; counts[("tp_" if s in truth else "fp_") + s[2]] += 1
            for s in truth - guess:
                counts["fn"] += 1; counts["fn_" + s[2]] += 1
            total.update(counts)
            lines.append(json.dumps({"group": row["group"], "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"],
                                     **{k: v for k, v in counts.items() if "_" in k}}))
    open(options.out, "w").write("\n".join(lines) + "\n")
    f1 = lambda tp, fp, fn: 2 * tp / max(2 * tp + fp + fn, 1)
    print(f"{options.model} {options.split}{' drop ' + options.drop if options.drop else ''}: micro-F1 {f1(total['tp'], total['fp'], total['fn']):.4f}")
    for kind in ("PER", "ORG", "LOC", "MISC"):
        print(f"  {kind:5} F1 {f1(total['tp_' + kind], total['fp_' + kind], total['fn_' + kind]):.4f}  "
              f"tp {total['tp_' + kind]} fp {total['fp_' + kind]} fn {total['fn_' + kind]}")


if __name__ == "__main__":
    main()
