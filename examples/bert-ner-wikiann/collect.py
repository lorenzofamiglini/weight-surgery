"""Step 3 input: the vectors that feed the head, one per word (its first sub-word), with the gold tag
id and the group, saved as FEATURES.pt for probe_refit.py.

Run: python collect.py validation features.pt
"""

import sys

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import model, split

name, out = sys.argv[1], sys.argv[2]
tokenizer = AutoTokenizer.from_pretrained(model())
network = AutoModelForTokenClassification.from_pretrained(model()).eval()
ids = network.config.label2id
rows = split(name)
hidden, gold, groups = [], [], []
for start in range(0, len(rows), 64):
    batch = rows[start:start + 64]
    encoded = tokenizer([r["words"] for r in batch], is_split_into_words=True, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        vectors = network.bert(**encoded).last_hidden_state
    for b, row in enumerate(batch):
        seen = set()
        for position, word in enumerate(encoded.word_ids(b)):
            if word is not None and word not in seen:
                seen.add(word)
                hidden.append(vectors[b, position]); gold.append(ids[row["tags"][word]]); groups.append(row["group"])
torch.save({"hidden": torch.stack(hidden), "gold": torch.tensor(gold), "groups": groups}, out)
print(f"{len(gold)} words from {len(rows)} sentences")
