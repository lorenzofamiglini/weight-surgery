"""Load and save a model directory's safetensors, shard by shard, and read its class names."""

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def shards(directory: str) -> dict[Path, dict[str, torch.Tensor]]:
    return {path: load_file(path) for path in sorted(Path(directory).glob("*.safetensors"))}


def tensor(directory: str, name: str) -> torch.Tensor:
    for values in shards(directory).values():
        if name in values:
            return values[name]
    raise KeyError(f"{name} not in {directory}")


def write(source: str, target: str, changed: dict[str, torch.Tensor]) -> None:
    Path(target).mkdir(parents=True, exist_ok=True)
    for path in Path(source).iterdir():
        if path.is_file() and path.suffix != ".safetensors":
            shutil.copy(path, Path(target) / path.name)
    for path, values in shards(source).items():
        save_file({name: changed.get(name, value).contiguous() for name, value in values.items()},
                  str(Path(target) / path.name), metadata={"format": "pt"})


def tag_names(directory: str, count: int) -> list[str]:
    config = Path(directory) / "config.json"
    labels = json.loads(config.read_text()).get("id2label", {}) if config.exists() else {}
    return [labels.get(str(i), str(i)) for i in range(count)]


def class_of(tag: str) -> str:
    prefix, _, rest = tag.partition("-")
    return rest if rest and prefix in {"B", "I", "E", "S", "L", "U"} else tag
