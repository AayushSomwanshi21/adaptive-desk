"""On-disk model versions with an "active" pointer.

Layout under the registry root:

    registry.json                  {"active": "v0003", "history": [...]}
    versions/
        v0001/
            model_state.pt         weights
            label_to_id.json       category -> output index (append-only)
            memory/                replay buffer (see ClassBalancedMemory.save)
            eval_set.jsonl         held-out labelled tickets, never trained on;
                                   grows with every version (gate + calibration)
            calibration.json       temperature + review threshold
            metrics.json           evaluation results for this version
            manifest.json          name, parent, notes, created_at

Promoting or rolling back a version only changes the pointer in
registry.json, never the version folders, so both are instant and safe.
"""

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

MODEL_FILE = "model_state.pt"
LABELS_FILE = "label_to_id.json"
MEMORY_DIR = "memory"
EVAL_SET_FILE = "eval_set.jsonl"
CALIBRATION_FILE = "calibration.json"
METRICS_FILE = "metrics.json"
MANIFEST_FILE = "manifest.json"

# A version without these cannot be served.
REQUIRED_FILES = (MODEL_FILE, LABELS_FILE)

_VERSION_PATTERN = re.compile(r"^v(\d{4,})$")


def write_json_atomic(path, data):
    """Write JSON so a reader never sees a half-written file."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    os.replace(temporary, path)


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


class ModelRegistry:
    def __init__(self, root):
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.state_path = self.root / "registry.json"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    # ---- state --------------------------------------------------------------
    def _read_state(self):
        if not self.state_path.exists():
            return {"active": None, "history": []}
        return read_json(self.state_path)

    def active(self):
        """Name of the version currently being served, or None."""
        return self._read_state()["active"]

    def active_dir(self):
        name = self.active()
        return self.version_dir(name) if name else None

    def history(self):
        """Versions in the order they were promoted (oldest first)."""
        return list(self._read_state()["history"])

    # ---- versions -----------------------------------------------------------
    def list_versions(self):
        names = [
            path.name for path in self.versions_dir.iterdir()
            if path.is_dir() and _VERSION_PATTERN.match(path.name)
        ]
        return sorted(names)

    def version_dir(self, name):
        path = self.versions_dir / name
        if not path.is_dir():
            raise FileNotFoundError(f"Unknown model version: {name}")
        return path

    def is_complete(self, name):
        path = self.versions_dir / name
        return path.is_dir() and all(
            (path / required).exists() for required in REQUIRED_FILES
        )

    def create_version(self, parent=None, notes=""):
        """Allocate the next version folder and write its manifest."""
        numbers = [
            int(_VERSION_PATTERN.match(name).group(1))
            for name in self.list_versions()
        ]
        name = f"v{(max(numbers) + 1) if numbers else 1:04d}"

        path = self.versions_dir / name
        path.mkdir(parents=True, exist_ok=False)  # fails loudly on a clash

        write_json_atomic(path / MANIFEST_FILE, {
            "name": name,
            "parent": parent,
            "notes": notes,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        return name, path

    def read_manifest(self, name):
        return read_json(self.version_dir(name) / MANIFEST_FILE)

    def read_metrics(self, name):
        path = self.version_dir(name) / METRICS_FILE
        return read_json(path) if path.exists() else {}

    # ---- calibration --------------------------------------------------------
    def write_calibration(self, name, calibration):
        write_json_atomic(
            self.version_dir(name) / CALIBRATION_FILE, calibration)

    def read_calibration(self, name):
        path = self.version_dir(name) / CALIBRATION_FILE
        return read_json(path) if path.exists() else {}

    # ---- promotion ----------------------------------------------------------
    def promote(self, name):
        """Make `name` the served version. Refuses incomplete versions."""
        if not self.is_complete(name):
            raise ValueError(
                f"Version {name} is missing required files "
                f"{list(REQUIRED_FILES)} and cannot be promoted.")

        state = self._read_state()
        if state["active"] != name:
            state["history"].append(name)
            state["active"] = name
            write_json_atomic(self.state_path, state)
        return name

    def rollback(self):
        """Go back to the previously promoted version."""
        state = self._read_state()
        if len(state["history"]) < 2:
            raise RuntimeError("No earlier version to roll back to.")

        state["history"].pop()
        state["active"] = state["history"][-1]
        write_json_atomic(self.state_path, state)
        return state["active"]

    # ---- import -------------------------------------------------------------
    def import_checkpoint(self, checkpoint_dir, memory=None, parent=None,
                          notes=""):
        """Turn a checkpoint written by the experiment runner into a version.

        Copies the weights, label map and metrics, and saves `memory` (a
        ClassBalancedMemory) if given. Handy to get a first servable version
        out of an existing experiment.
        """
        checkpoint_dir = Path(checkpoint_dir)
        for required in REQUIRED_FILES:
            if not (checkpoint_dir / required).exists():
                raise FileNotFoundError(
                    f"{checkpoint_dir} has no {required}; save the "
                    "checkpoint with weights (SAVE_STAGE_CHECKPOINTS=True).")

        name, path = self.create_version(parent=parent, notes=notes)
        for filename in (MODEL_FILE, LABELS_FILE, METRICS_FILE):
            source = checkpoint_dir / filename
            if source.exists():
                shutil.copy2(source, path / filename)

        if memory is not None:
            memory.save(path / MEMORY_DIR)
        return name
