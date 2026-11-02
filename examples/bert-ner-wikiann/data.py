"""Download dslim/bert-base-NER (safetensors only) and WikiANN English, and read a split as words, gold
tags in the model's tag names, and a group per block of 20 sentences (WikiANN has no documents)."""

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

HERE = Path(__file__).parent
MODEL = HERE / "model"
WIKIANN = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def model() -> str:
    if not (MODEL / "model.safetensors").exists():
        snapshot_download("dslim/bert-base-NER", local_dir=MODEL,
                          allow_patterns=["config.json", "model.safetensors", "vocab.txt", "*token*.json"])
    return str(MODEL)


def split(name: str) -> list[dict]:
    path = hf_hub_download("unimelb-nlp/wikiann", f"en/{name}/0000.parquet", repo_type="dataset",
                           revision="refs/convert/parquet", local_dir=HERE / "data")
    frame = pd.read_parquet(path)
    return [{"words": list(row.tokens), "tags": [WIKIANN[t] for t in row.ner_tags], "group": f"{name}-{i // 20}"}
            for i, row in enumerate(frame.itertuples())]
