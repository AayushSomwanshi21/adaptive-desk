
import json
import random
from pathlib import Path

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

TRAIN_PATH = "ml/datasets/train.csv"
TEST_PATH = "ml/datasets/test.csv"
MODEL_NAME = "distilbert-base-uncased"

STAGE1_DIR = Path("ml/models/checkpoints/stage1_baseline")
OUTPUT_DIR = Path("ml/models/checkpoints/stage2_baseline")


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


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load and validate the datasets.
    train_df = validate_dataset(
        load_dataset(TRAIN_PATH), remove_duplicates=True
    )
    test_df = validate_dataset(
        load_dataset(TEST_PATH), remove_duplicates=False
    )

    # Recreate exactly the same incremental category order as Stage 1.
    stages = create_incremental_splits(
        train_df,
        test_df,
        classes_per_stage=10,
        seed=SEED,
    )

    stage = stages[1]  # Stage 2
    print("\nTraining Stage:", stage["stage"])
    print("New categories:", len(stage["new_classes"]))
    print("Previously seen categories:", len(stages[0]["seen_classes"]))
    print("Total categories:", len(stage["seen_classes"]))
    print("Training examples:", len(stage["train"]))
    print("Test examples:", len(stage["test"]))

    # Load the Stage 1 label mapping.
    with open(STAGE1_DIR / "label_to_id.json", encoding="utf-8") as file:
        old_label_to_id = json.load(file)

    # Preserve old category IDs and append IDs for new categories.
    label_to_id = {
        label: idx for idx, label in enumerate(stage["seen_classes"])
    }

    for label, old_id in old_label_to_id.items():
        if label_to_id.get(label) != old_id:
            raise ValueError(
                f"Category ID changed for {label}: "
                f"expected {old_id}, got {label_to_id.get(label)}"
            )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = prepare_dataset(
        stage["train"], tokenizer, label_to_id
    )
    test_dataset = prepare_dataset(
        stage["test"], tokenizer, label_to_id
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # Reconstruct the Stage 1 model and load its trained weights.
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

    # Expand the classifier from 10 outputs to 20.
    model.expand_classifier(len(label_to_id))
    model = model.to(device)

    # Create the optimizer AFTER expanding the classifier.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Train only on the newly introduced categories.
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

        average_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Training loss: {average_loss:.4f}"
        )

    # Evaluate across all 20 categories seen so far.
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

    print("\n=== Stage 2 Evaluation ===")
    metrics = calculate_metrics(
        all_labels,
        all_predictions,
        class_names=stage["seen_classes"],
    )

    # Separate performance on old and newly introduced categories.
    all_labels_array = np.array(all_labels)
    all_predictions_array = np.array(all_predictions)

    old_mask = all_labels_array < len(old_label_to_id)
    new_mask = ~old_mask

    old_accuracy = accuracy_score(
        all_labels_array[old_mask],
        all_predictions_array[old_mask],
    )

    new_accuracy = accuracy_score(
        all_labels_array[new_mask],
        all_predictions_array[new_mask],
    )

    print("\n=== Old vs New Category Performance ===")
    print(f"Old-category accuracy: {old_accuracy:.4f}")
    print(f"New-category accuracy: {new_accuracy:.4f}")

    print(f"Old-category accuracy (%): {old_accuracy * 100:.2f}%")
    print(f"New-category accuracy (%): {new_accuracy * 100:.2f}%")
    print(f"Stage 1 baseline accuracy (%): 99.00%")
    print(
        "Old-category accuracy drop (percentage points): "
        f"{99.00 - old_accuracy * 100:.2f}"
    )

    # Save the Stage 2 checkpoint separately from Stage 1.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(), OUTPUT_DIR / "model_state.pt"
    )
    model.backbone.config.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(
        OUTPUT_DIR / "label_to_id.json", "w", encoding="utf-8"
    ) as file:
        json.dump(label_to_id, file, indent=2)

    with open(
        OUTPUT_DIR / "metrics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    print("\nCheckpoint saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
