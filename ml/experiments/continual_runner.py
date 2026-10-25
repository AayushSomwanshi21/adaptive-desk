
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from ml.datasets.incremental_splitter import create_incremental_splits
from ml.datasets.loader import load_dataset
from ml.datasets.validator import validate_dataset
from ml.evaluation.metrics import calculate_metrics
from ml.models.text_classifier import AdaptiveDeskClassifier


ROOT = Path.cwd()
TRAIN_PATH = ROOT / "ml/datasets/train.csv"
TEST_PATH = ROOT / "ml/datasets/test.csv"
CHECKPOINT_ROOT = ROOT / "ml/models/checkpoints"

SEED = 42
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
MEMORY_PER_CLASS = 20
MODEL_NAME = "distilbert-base-uncased"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    """Load and validate the dataset using the shared ML loader helpers."""

    train_df = validate_dataset(
        load_dataset(str(TRAIN_PATH)),
        remove_duplicates=True,
    )
    test_df = validate_dataset(
        load_dataset(str(TEST_PATH)),
        remove_duplicates=False,
    )

    return train_df, test_df


def prepare_dataset(dataframe, tokenizer, label_to_id):
    labels = dataframe["label"].map(label_to_id)

    if labels.isna().any():
        raise ValueError(
            "Dataset contains a label missing from the label mapping."
        )

    encodings = tokenizer(
        dataframe["text"].tolist(),
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


def make_loader(dataframe, label_to_id, tokenizer, shuffle):
    dataset = prepare_dataset(dataframe, tokenizer, label_to_id)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def build_replay_memory(train_df, old_classes, stage_number):
    """Sample up to 20 training examples per previously seen class."""

    rng = np.random.default_rng(SEED + stage_number)
    memory_parts = []

    for category in old_classes:
        category_df = train_df[
            train_df["label"] == category
        ]

        count = min(MEMORY_PER_CLASS, len(category_df))
        if count == 0:
            continue

        random_state = int(rng.integers(0, 2**31 - 1))
        memory_parts.append(
            category_df.sample(n=count, random_state=random_state)
        )

    if not memory_parts:
        raise ValueError("No examples were selected for replay memory.")

    return pd.concat(memory_parts, ignore_index=True)


def train_model(model, loader):
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            # set gradients to zero
            optimizer.zero_grad()

            # get the logits of the text classifier (DistilBERT)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # calculate the loss
            loss = criterion(logits, labels)

            # backpropogation and calculate the gradient
            loss.backward()
            # update the weights
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / max(len(loader), 1)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Training loss: {average_loss:.4f}"
        )


def evaluate_model(model, test_df, label_to_id, tokenizer,
                   old_classes, new_classes):
    loader = make_loader(
        test_df, label_to_id, tokenizer, shuffle=False
    )

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            logits = model(
                input_ids=input_ids.to(DEVICE),
                attention_mask=attention_mask.to(DEVICE),
            )
            # selects the index of the max prediction
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)

    ordered_labels = [
        label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])
    ]
    summary_metrics = calculate_metrics(
        y_true,
        y_pred,
        class_names=ordered_labels,
    )

    old_ids = {label_to_id[c] for c in old_classes}
    new_ids = {label_to_id[c] for c in new_classes}

    old_mask = np.isin(y_true, list(old_ids))
    new_mask = np.isin(y_true, list(new_ids))

    metrics = {
        "accuracy": float(summary_metrics["accuracy"]),
        "macro_f1": float(summary_metrics["macro_f1"]),
        "old_category_accuracy": (
            float(accuracy_score(y_true[old_mask], y_pred[old_mask]))
            if old_mask.any() else None
        ),
        "new_category_accuracy": (
            float(accuracy_score(y_true[new_mask], y_pred[new_mask]))
            if new_mask.any() else None
        ),
        "test_examples": int(len(y_true)),
        "old_categories": int(len(old_classes)),
        "new_categories": int(len(new_classes)),
    }

    return metrics


def run_experiment(strategy):
    if strategy not in {"baseline", "replay"}:
        raise ValueError("strategy must be 'baseline' or 'replay'")

    set_seed()
    print(f"Using device: {DEVICE}")
    print(f"Strategy: {strategy}")

    # load training and test data
    train_df, test_df = load_data()

    # create incremental splits
    stages = create_incremental_splits(
        train_df,
        test_df,
        classes_per_stage=10,
        seed=SEED,
    )

    # Continue from the Stage 3 checkpoint already trained for this method.
    previous_stage = 3
    previous_dir = CHECKPOINT_ROOT / f"stage{previous_stage}_{strategy}"

    if not previous_dir.exists():
        raise FileNotFoundError(
            f"Missing checkpoint directory: {previous_dir}"
        )

    mapping_path = previous_dir / "label_to_id.json"
    weights_path = previous_dir / "model_state.pt"

    with open(mapping_path, "r", encoding="utf-8") as file:
        label_to_id = json.load(file)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AdaptiveDeskClassifier(
        num_labels=len(label_to_id)
    )
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    model.to(DEVICE)

    results = []

    # Starting from stage 4 since Stage 1, 2, 3 already trained.
    # due to 0 based indexing stage_index 3 represents stage 4
    for stage_index in range(3, len(stages)):
        # load the current stage
        stage = stages[stage_index]
        stage_number = stage_index + 1

        old_classes = list(stages[stage_index - 1]["seen_classes"])
        new_classes = list(stage["new_classes"])
        # total categories till now (old + current)
        seen_classes = list(stage["seen_classes"])

        print("\n" + "=" * 60)
        print(f"Stage {stage_number}")
        print(f"Previously seen categories: {len(old_classes)}")
        print(f"New categories: {len(new_classes)}")
        print(f"Total categories: {len(seen_classes)}")

        stage_train = stage["train"].copy()
        stage_test = stage["test"].copy()

        # Add new categories without changing existing label IDs.
        for category in new_classes:
            category = str(category)
            if category not in label_to_id:
                label_to_id[category] = len(label_to_id)

        if set(label_to_id) != {str(c) for c in seen_classes}:
            raise ValueError(
                f"Label mapping does not match Stage {stage_number} classes."
            )

        if len(label_to_id) != len(seen_classes):
            raise ValueError("Duplicate or missing label IDs detected.")

        # Expand classifier while preserving existing output weights.
        model.expand_classifier(len(label_to_id))
        model.to(DEVICE)

        replay_count = 0

        if strategy == "replay":
            memory_df = build_replay_memory(
                train_df, old_classes, stage_number
            )
            replay_count = len(memory_df)

            # combine the current training data with the replay data
            stage_train = pd.concat(
                [stage_train, memory_df],
                ignore_index=True,
            )

            print(f"Replay examples: {replay_count}")

        print(f"Training examples: {len(stage_train)}")
        print(f"Test examples: {len(stage_test)}")

        train_loader = make_loader(
            stage_train, label_to_id, tokenizer, shuffle=True
        )

        train_model(model, train_loader)

        metrics = evaluate_model(
            model,
            stage_test,
            label_to_id,
            tokenizer,
            old_classes,
            new_classes,
        )

        metrics.update({
            "stage": stage_number,
            "strategy": strategy,
            "training_examples": int(len(stage_train)),
            "replay_examples": int(replay_count),
        })

        print("\nEvaluation")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro-F1: {metrics['macro_f1']:.4f}")
        print(
            "Old-category accuracy: "
            f"{metrics['old_category_accuracy']:.4f}"
        )
        print(
            "New-category accuracy: "
            f"{metrics['new_category_accuracy']:.4f}"
        )

        output_dir = CHECKPOINT_ROOT / f"stage{stage_number}_{strategy}"
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), output_dir / "model_state.pt")
        tokenizer.save_pretrained(output_dir)

        with open(
            output_dir / "label_to_id.json", "w", encoding="utf-8"
        ) as file:
            json.dump(label_to_id, file, indent=2)

        with open(
            output_dir / "metrics.json", "w", encoding="utf-8"
        ) as file:
            json.dump(metrics, file, indent=2)

        results.append(metrics)

        # This model now becomes the starting model for the next stage.
        previous_dir = output_dir
        print(f"Checkpoint saved to: {output_dir}")

    results_dir = ROOT / "ml" / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"results_{strategy}_stages4_8.csv"

    pd.DataFrame(results).to_csv(results_path, index=False)

    print(f"\nAll remaining stages completed.")
    print(f"Results saved to: {results_path}")
