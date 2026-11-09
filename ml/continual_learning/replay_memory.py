"""Class-balanced replay memory for class-incremental learning."""

import numpy as np
import pandas as pd


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

            # Random priority order
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
