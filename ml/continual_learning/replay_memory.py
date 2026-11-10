"""Class-balanced replay memory for class-incremental learning."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

EXEMPLARS_FILE = "memory_exemplars.jsonl"
META_FILE = "memory_meta.json"


class ClassBalancedMemory:
    """Fixed-capacity, persistent, class-balanced replay buffer.

    - The buffer only ever holds exemplars that were stored when a class was
      first learned. It never reads old training data again afterwards.
    - Total size is capped at `capacity`; every seen class gets an equal quota.
    - When new classes arrive, old classes are shrunk (never regrown) so that
      every class, old and new, keeps a protected share.
    - Exemplars of each class are kept in a random priority order, so shrinking
      a class just means dropping the tail of its list.
    """

    def __init__(self, capacity, seed=42):
        self.capacity = capacity
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.slots = {}  # class name (str) -> DataFrame of exemplars

    def _quotas(self):
        """Per-class sizes: equal split, with the remainder spread out."""
        num_classes = len(self.slots)
        if num_classes == 0:
            return {}
        base = max(1, self.capacity // num_classes)
        extra = max(0, self.capacity - base * num_classes)
        return {
            category: base + (1 if index < extra else 0)
            for index, category in enumerate(self.slots)
        }

    def update(self, stage_df, new_classes):
        """Register new classes from the CURRENT stage's data, then rebalance."""
        labels = stage_df["label"].astype(str)

        for category in new_classes:
            category = str(category)
            category_df = stage_df[labels == category]
            if len(category_df) == 0:
                continue

            # Random priority order (swap for herding if you have embeddings).
            order = self.rng.permutation(len(category_df))
            self.slots[category] = category_df.iloc[order].reset_index(
                drop=True
            )

        # Shrink every class (old and new) to its quota.
        for category, quota in self._quotas().items():
            self.slots[category] = self.slots[category].iloc[:quota]

    def as_dataframe(self):
        if not self.slots:
            raise ValueError("Replay memory is empty.")
        return pd.concat(self.slots.values(), ignore_index=True)

    def summary(self):
        sizes = [len(df) for df in self.slots.values()]
        return (
            f"{len(self.slots)} classes, {sum(sizes)}/{self.capacity} slots, "
            f"{min(sizes)}-{max(sizes)} per class"
        )

    def save(self, directory):
        """Write the buffer to `directory` so it survives restarts.

        Two files: the exemplars (JSON lines, one class after another, each
        class in its priority order) and a small metadata file holding the
        capacity, the seed and the class order. Class order matters: the
        earliest classes get the spare slots when the capacity does not divide
        evenly, so it must be restored exactly.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        meta = {
            "capacity": self.capacity,
            "seed": self.seed,
            "class_order": list(self.slots),
        }
        with open(directory / META_FILE, "w", encoding="utf-8") as file:
            json.dump(meta, file, indent=2)

        exemplars_path = directory / EXEMPLARS_FILE
        if self.slots:
            self.as_dataframe().to_json(
                exemplars_path, orient="records", lines=True,
                force_ascii=False,
            )
        else:
            exemplars_path.write_text("", encoding="utf-8")

    @classmethod
    def load(cls, directory):
        """Rebuild a buffer written by `save`."""
        directory = Path(directory)
        meta_path = directory / META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"No replay memory found in {directory}")

        with open(meta_path, "r", encoding="utf-8") as file:
            meta = json.load(file)

        memory = cls(meta["capacity"], seed=meta["seed"])
        # A fresh stream for future random selections; seeded by how many
        # classes are already stored so versions do not reuse the same one.
        memory.rng = np.random.default_rng(
            [meta["seed"], len(meta["class_order"])]
        )

        exemplars_path = directory / EXEMPLARS_FILE
        if meta["class_order"]:
            # dtype=False keeps text exactly as written (e.g. "007" stays a
            # string); labels are forced to str to match the rest of the code.
            frame = pd.read_json(
                exemplars_path, lines=True, dtype=False, convert_dates=False
            )
            frame["label"] = frame["label"].astype(str)

            for category in meta["class_order"]:
                memory.slots[category] = frame[
                    frame["label"] == category
                ].reset_index(drop=True)

        return memory
