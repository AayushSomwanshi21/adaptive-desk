
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
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

STAGE2_DIR = Path("ml/models/checkpoints/stage2_replay")
OUTPUT_DIR = Path("ml/models/checkpoints/stage3_replay")


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
    """Sample up to MEMORY_PER_CLASS examples per old category."""
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

    if not memory_parts:
        raise ValueError("No examples were selected for replay memory.")

    return pd.concat(memory_parts, ignore_index=True)


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    # Load and validate the datasets.
    train_df = validate_dataset(
        load_dataset(TRAIN_PATH),
        remove_duplicates=True,
    )
    test_df = validate_dataset(
        load_dataset(TEST_PATH),
        remove_duplicates=False,
    )

    # Recreate the same incremental category order used previously.
    stages = create_incremental_splits(
        train_df,
        test_df,
        classes_per_stage=10,
        seed=SEED,
    )

    stage2 = stages[1]
    stage3 = stages[2]

    old_classes = stage2["seen_classes"]  # 20 old categories
    new_classes = stage3["new_classes"]   # 10 new categories
    seen_classes = stage3["seen_classes"]  # 30 categories in total

    print("\nTraining Stage: 3 — Experience Replay")
    print("Previously seen categories:", len(old_classes))
    print("New categories:", len(new_classes))
    print("Total categories:", len(seen_classes))
    print("New training examples:", len(stage3["train"]))
    print("Evaluation examples:", len(stage3["test"]))

    # Load the Stage 2 replay mapping.
    with open(
        STAGE2_DIR / "label_to_id.json",
        encoding="utf-8",
    ) as file:
        old_label_to_id = json.load(file)

    # Preserve old IDs and append IDs for the new categories.
    label_to_id = old_label_to_id.copy()

    for label in new_classes:
        if label not in label_to_id:
            label_to_id[label] = len(label_to_id)

    if set(label_to_id) != set(seen_classes):
        raise ValueError(
            "Label mapping does not match the 30 categories seen so far."
        )

    if len(label_to_id) != 30:
        raise ValueError(
            f"Expected 30 category IDs, got {len(label_to_id)}."
        )

    # Select replay examples only from the training data.
    old_train_df = train_df[
        train_df["label"].isin(old_classes)
    ]

    memory_df = build_replay_memory(
        old_train_df,
        old_classes,
    )

    new_train_df = stage3["train"]

    replay_train_df = pd.concat(
        [new_train_df, memory_df],
        ignore_index=True,
    )

    print("Old replay examples:", len(memory_df))
    print("Combined training examples:", len(replay_train_df))

    # Prepare tokenized datasets.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = prepare_dataset(
        replay_train_df,
        tokenizer,
        label_to_id,
    )
    test_dataset = prepare_dataset(
        stage3["test"],
        tokenizer,
        label_to_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Load the Stage 2 replay model, which already has 20 outputs.
    model = AdaptiveDeskClassifier(
        num_labels=len(old_label_to_id),
        model_name=MODEL_NAME,
    ).to(device)

    state = torch.load(
        STAGE2_DIR / "model_state.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)

    # Expand the classifier from 20 outputs to 30.
    model.expand_classifier(len(label_to_id))
    model = model.to(device)

    # Initialize optimizer after expanding the classifier.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Train on new examples and replay examples together.
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
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Training loss: {average_loss:.4f}"
        )

    # Evaluate across all 30 categories.
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

    print("\n=== Stage 3 Replay Evaluation ===")

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
        class_names=seen_classes,
    )

    # Separate accuracy for old and new categories.
    true_array = np.array(all_labels)
    pred_array = np.array(all_predictions)

    old_label_ids = {
        label_to_id[label] for label in old_classes
    }

    old_mask = np.isin(
        true_array,
        list(old_label_ids),
    )
    new_mask = ~old_mask

    old_accuracy = accuracy_score(
        true_array[old_mask],
        pred_array[old_mask],
    )
    new_accuracy = accuracy_score(
        true_array[new_mask],
        pred_array[new_mask],
    )

    metrics["old_category_accuracy"] = float(old_accuracy)
    metrics["new_category_accuracy"] = float(new_accuracy)

    print(
        f"\nOld-category accuracy: "
        f"{old_accuracy:.4f} ({old_accuracy:.2%})"
    )
    print(
        f"New-category accuracy: "
        f"{new_accuracy:.4f} ({new_accuracy:.2%})"
    )

    # Save the Stage 3 replay checkpoint and results.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        OUTPUT_DIR / "model_state.pt",
    )
    model.backbone.config.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(
        OUTPUT_DIR / "label_to_id.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(label_to_id, file, indent=2)

    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    print("\nCheckpoint saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
