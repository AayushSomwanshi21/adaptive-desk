import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

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

OUTPUT_DIR = Path("ml/models/checkpoints/stage1_baseline")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataset(df, tokenizer, label_to_id):
    labels = df["label"].map(label_to_id)

    if labels.isna().any():
        raise ValueError(
            "Dataset contains a label missing from the label mapping.")

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

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    # Load and validate the training and test data.
    train_df = validate_dataset(
        load_dataset(TRAIN_PATH),
        remove_duplicates=True,
    )

    test_df = validate_dataset(
        load_dataset(TEST_PATH),
        remove_duplicates=False,
    )

    # Create the class-incremental learning stages.
    stages = create_incremental_splits(
        train_df,
        test_df,
        classes_per_stage=10,
        seed=SEED,
    )

    stage = stages[0]

    print("\nTraining Stage:", stage["stage"])
    print("New categories:", len(stage["new_classes"]))
    print("Training examples:", len(stage["train"]))
    print("Test examples:", len(stage["test"]))

    # Keep the class-to-ID mapping consistent.
    classes = stage["seen_classes"]
    label_to_id = {
        label: idx for idx, label in enumerate(classes)
    }

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = prepare_dataset(
        stage["train"], tokenizer, label_to_id
    )
    test_dataset = prepare_dataset(
        stage["test"], tokenizer, label_to_id
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

    model = AdaptiveDeskClassifier(
        num_labels=len(classes),
        model_name=MODEL_NAME,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = torch.nn.CrossEntropyLoss()

    # Train the Stage 1 baseline.
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # reset the gradient for each batch
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

    # Evaluate on the categories introduced in Stage 1.
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(input_ids, attention_mask)
            # take the predication index with the highest confidence
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    print("\n=== Stage 1 Evaluation ===")
    metrics = calculate_metrics(
        all_labels,
        all_predictions,
        class_names=classes,
    )

    # Save the checkpoint and the exact label mapping.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        OUTPUT_DIR / "model_state.pt",
    )

    model.backbone.config.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(OUTPUT_DIR / "label_to_id.json", "w", encoding="utf-8") as file:
        json.dump(label_to_id, file, indent=2)

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nCheckpoint saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
