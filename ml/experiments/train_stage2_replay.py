
import json
import random
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score

from ml.datasets.loader import load_dataset
from ml.datasets.validator import validate_dataset
from ml.datasets.incremental_splitter import create_incremental_splits
from ml.models.text_classifier import AdaptiveDeskClassifier
from ml.evaluation.metrics import calculate_metrics


SEED = 42
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
MEMORY_PER_CLASS = 20

TRAIN_PATH = "ml/datasets/train.csv"
TEST_PATH = "ml/datasets/test.csv"
MODEL_NAME = "distilbert-base-uncased"

STAGE1_DIR = Path("ml/models/checkpoints/stage1_baseline")
OUTPUT_DIR = Path("ml/models/checkpoints/stage2_replay")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataset(df, tokenizer, label_to_id):
    labels = df["label"].map(label_to_id)

    if labels.isna().any():
        raise ValueError("A dataset label is missing from the label mapping.")

    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    return TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        torch.tensor(labels.to_numpy(), dtype=torch.long),
    )


def build_replay_memory(old_train_df, old_classes):
    """Select a fixed number of examples from each old category."""
    rng = np.random.default_rng(SEED)
    memory_parts = []

    for category in old_classes:
        category_df = old_train_df[
            old_train_df["label"] == category
        ]

        sample_count = min(MEMORY_PER_CLASS, len(category_df))
        random_state = int(rng.integers(0, 2**31 - 1))

        sampled = category_df.sample(
            n=sample_count,
            random_state=random_state,
        )

        memory_parts.append(sampled)

    return pd.concat(memory_parts, ignore_index=True)


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_df = validate_dataset(
        load_dataset(TRAIN_PATH), remove_duplicates=True
    )
    test_df = validate_dataset(
        load_dataset(TEST_PATH), remove_duplicates=False
    )

    stages = create_incremental_splits(
        train_df,
        test_df,
        classes_per_stage=10,
        seed=SEED,
    )

    stage1 = stages[0]
    stage2 = stages[1]

    old_classes = stage1["seen_classes"]
    new_classes = stage2["new_classes"]
    seen_classes = stage2["seen_classes"]

    print("\nTraining Stage: 2 — Experience Replay")
    print("Old categories:", len(old_classes))
    print("New categories:", len(new_classes))

    # Preserve the exact category IDs established at Stage 1.
    with open(STAGE1_DIR / "label_to_id.json", encoding="utf-8") as file:
        old_label_to_id = json.load(file)

    label_to_id = {
        label: idx for idx, label in enumerate(seen_classes)
    }

    for label, old_id in old_label_to_id.items():
        if label_to_id.get(label) != old_id:
            raise ValueError(f"Category ID changed for {label}")

    # Select old examples from the training data only.
    old_train_df = train_df[
        train_df["label"].isin(old_classes)
    ]

    # Select 20 old examples per category from training data only.
    memory_df = build_replay_memory(old_train_df, old_classes)
    new_train_df = stage2["train"]

    # Combine the old memory with the new-category training data.
    replay_train_df = pd.concat(
        [new_train_df, memory_df],
        ignore_index=True,
    )

    print("Old replay examples:", len(memory_df))
    print("New training examples:", len(stage2["train"]))
    print("Combined training examples:", len(replay_train_df))
    print("Evaluation examples:", len(stage2["test"]))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = prepare_dataset(
        replay_train_df, tokenizer, label_to_id
    )
    test_dataset = prepare_dataset(
        stage2["test"], tokenizer, label_to_id
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # Start from the same Stage 1 checkpoint as the baseline.
    model = AdaptiveDeskClassifier(
        num_labels=len(old_label_to_id),
        model_name=MODEL_NAME,
    ).to(device)

    state = torch.load(
        STAGE1_DIR / "model_state.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)

    model.expand_classifier(len(label_to_id))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE
    )
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Training loss: {total_loss / len(train_loader):.4f}"
        )

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    print("\n=== Stage 2 Replay Evaluation ===")
    metrics = calculate_metrics(
        all_labels,
        all_predictions,
        class_names=seen_classes,
    )

    true_array = np.array(all_labels)
    pred_array = np.array(all_predictions)
    old_label_ids = {
        label_to_id[label] for label in old_classes
    }

    old_mask = np.isin(true_array, list(old_label_ids))
    new_mask = ~old_mask

    old_accuracy = accuracy_score(
        true_array[old_mask], pred_array[old_mask]
    )
    new_accuracy = accuracy_score(
        true_array[new_mask], pred_array[new_mask]
    )

    metrics["old_category_accuracy"] = float(old_accuracy)
    metrics["new_category_accuracy"] = float(new_accuracy)

    print(f"\nOld-category accuracy: {old_accuracy:.4f} ({old_accuracy:.2%})")
    print(f"New-category accuracy: {new_accuracy:.4f} ({new_accuracy:.2%})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_DIR / "model_state.pt")
    model.backbone.config.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(OUTPUT_DIR / "label_to_id.json", "w", encoding="utf-8") as file:
        json.dump(label_to_id, file, indent=2)

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nCheckpoint saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
