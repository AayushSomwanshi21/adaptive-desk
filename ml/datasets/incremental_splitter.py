import numpy as np
import pandas as pd


def create_incremental_splits(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        classes_per_stage: int = 10,
        seed: int = 42
):
    """
    Create class-incremental learning stages.
    Each stage contains training examples for newly introduced
    categories and test examples for all categories seen so far.
    """

    if classes_per_stage <= 0:
        raise ValueError("classes_per_stage must be positive")

    train_classes = set(train_df['label'].unique())
    test_classes = set(test_df['label'].unique())

    if train_classes != test_classes:
        raise ValueError(
            "Training and test datasets must have matching categories.")

    classes = sorted(train_classes)

    rng = np.random.default_rng(seed)
    rng.shuffle(classes)

    stages = []
    seen_classes = []

    for start in range(0, len(classes), classes_per_stage):

        new_classes = classes[start: start + classes_per_stage]
        seen_classes.extend(new_classes)

        # Only train on the categories introduced in this stage
        stage_train = train_df[train_df['label'].isin(new_classes)].copy()

        # Evaluate on every category learned so far.
        stage_test = test_df[test_df['label'].isin(seen_classes)].copy()

        stages.append({
            "stage": len(stages) + 1,
            "new_classes": list(new_classes),
            "seen_classes": list(seen_classes),
            "train": stage_train.reset_index(drop=True),
            "test": stage_test.reset_index(drop=True)
        })

    return stages
