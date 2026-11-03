"""The hand edit: give the MISC rows of the head the ORG label (WikiANN tags songs, leagues and shows,
which CoNLL calls MISC, as ORG). Copies the model and changes two entries of id2label; no number changes.

Run: python relabel.py MODEL_DIR OUT_DIR
"""

import json
import shutil
import sys
from pathlib import Path

source, target = Path(sys.argv[1]), Path(sys.argv[2])
shutil.copytree(source, target, dirs_exist_ok=True)
config = json.loads((target / "config.json").read_text())
config["id2label"] = {i: tag.replace("MISC", "ORG") for i, tag in config["id2label"].items()}
(target / "config.json").write_text(json.dumps(config, indent=2))
print(f"wrote {target}: MISC rows labelled ORG")
