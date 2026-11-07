
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from ml.datasets.loader import load_dataset
from ml.datasets.validator import validate_dataset
from ml.datasets.incremental_splitter import create_incremental_splits
from ml.models.text_classifier import AdaptiveDeskClassifier


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "ml/datasets/train.csv"
TEST_PATH = ROOT / "ml/datasets/test.csv"

CHECKPOINT_ROOT = ROOT / "ml/models/checkpoints"
RESULTS_DIR = ROOT / "ml/experiments/results"

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 8
MAX_LENGTH = 128
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_data():
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
            "Dataset contains a label missing from the mapping."
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


def make_loader(dataframe, label_to_id, tokenizer):
    dataset = prepare_dataset(
        dataframe,
        tokenizer,
        label_to_id,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_checkpoint(stage_number, strategy, test_df, tokenizer):
    # Stage 1 is the common initial model for both methods.
    checkpoint_strategy = (
        "baseline" if stage_number == 1 else strategy
    )

    checkpoint_dir = (
        CHECKPOINT_ROOT
        / f"stage{stage_number}_{checkpoint_strategy}"
    )

    weights_path = checkpoint_dir / "model_state.pt"
    mapping_path = checkpoint_dir / "label_to_id.json"

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_dir}"
        )

    with open(mapping_path, encoding="utf-8") as file:
        label_to_id = json.load(file)

    label_to_id = {
        label: int(label_id)
        for label, label_id in label_to_id.items()
    }

    # Verify classifier label IDs.
    label_ids = sorted(label_to_id.values())

    if label_ids != list(range(len(label_ids))):
        raise ValueError(
            f"Invalid label IDs in {mapping_path}"
        )

    model = AdaptiveDeskClassifier(
        num_labels=len(label_to_id),
        model_name=MODEL_NAME,
    )

    state = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    loader = make_loader(
        test_df,
        label_to_id,
        tokenizer,
    )

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(labels.tolist())

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)

    # Calculate accuracy separately for each category.
    category_results = []

    for category, category_id in label_to_id.items():
        category_mask = y_true == category_id

        if not category_mask.any():
            continue

        category_accuracy = accuracy_score(
            y_true[category_mask],
            y_pred[category_mask],
        )

        category_results.append({
            "stage": stage_number,
            "strategy": strategy,
            "category": category,
            "category_id": category_id,
            "accuracy": float(category_accuracy),
            "correct": int(
                (y_true[category_mask] == y_pred[category_mask]).sum()
            ),
            "total": int(category_mask.sum()),
        })

    overall_accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    print(
        f"Stage {stage_number} | {strategy} | "
        f"Accuracy: {overall_accuracy:.4f}"
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return category_results


# Forgetting calculation

def calculate_forgetting(category_accuracy_df):
    forgetting_rows = []

    for strategy in ["baseline", "replay"]:
        strategy_df = category_accuracy_df[
            category_accuracy_df["strategy"] == strategy
        ]

        for category, category_df in strategy_df.groupby("category"):
            category_df = category_df.sort_values("stage")

            best_previous_accuracy = None

            for row in category_df.itertuples():
                current_accuracy = row.accuracy

                if best_previous_accuracy is None:
                    forgetting = 0.0
                else:
                    forgetting = (
                        best_previous_accuracy - current_accuracy
                    )

                forgetting_rows.append({
                    "stage": row.stage,
                    "strategy": strategy,
                    "category": category,
                    "accuracy": current_accuracy,
                    "best_previous_accuracy": (
                        best_previous_accuracy
                    ),
                    "forgetting": float(forgetting),
                })

                # Update best accuracy observed so far.
                if best_previous_accuracy is None:
                    best_previous_accuracy = current_accuracy
                else:
                    best_previous_accuracy = max(
                        best_previous_accuracy,
                        current_accuracy,
                    )

    forgetting_df = pd.DataFrame(forgetting_rows)

    # Average forgetting across all categories seen at each stage.
    average_forgetting_df = (
        forgetting_df.groupby(["stage", "strategy"])
        ["forgetting"]
        .mean()
        .reset_index()
    )

    average_forgetting_df.rename(
        columns={"forgetting": "average_forgetting"},
        inplace=True,
    )

    return forgetting_df, average_forgetting_df


def main():
    torch.manual_seed(SEED)

    print("Using device:", DEVICE)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df, test_df = load_data()

    stages = create_incremental_splits(
        train_df,
        test_df,
        classes_per_stage=10,
        seed=SEED,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    category_results = []

    # Baseline: Stages 1-8
    # Replay: Stages 2-8
    for strategy in ["baseline", "replay"]:

        start_stage = 1 if strategy == "baseline" else 2

        for stage_number in range(
            start_stage,
            len(stages) + 1,
        ):
            stage = stages[stage_number - 1]

            print(
                f"\nEvaluating Stage {stage_number} "
                f"({strategy})"
            )

            results = evaluate_checkpoint(
                stage_number,
                strategy,
                stage["test"],
                tokenizer,
            )

            category_results.extend(results)

    # Save category-wise accuracy.
    category_accuracy_df = pd.DataFrame(
        category_results
    )

    category_accuracy_path = (
        RESULTS_DIR / "category_accuracy_all_stages.csv"
    )

    category_accuracy_df.to_csv(
        category_accuracy_path,
        index=False,
    )

    # Calculate forgetting.
    forgetting_df, average_forgetting_df = (
        calculate_forgetting(category_accuracy_df)
    )

    forgetting_path = (
        RESULTS_DIR / "forgetting_per_category.csv"
    )

    average_forgetting_path = (
        RESULTS_DIR / "average_forgetting_all_stages.csv"
    )

    forgetting_df.to_csv(
        forgetting_path,
        index=False,
    )

    average_forgetting_df.to_csv(
        average_forgetting_path,
        index=False,
    )

    print("\nEvaluation completed.")

    print("\nSaved:")
    print(category_accuracy_path)
    print(forgetting_path)
    print(average_forgetting_path)

    print("\nAverage Forgetting:")
    print(
        average_forgetting_df.pivot(
            index="stage",
            columns="strategy",
            values="average_forgetting",
        ).to_string()
    )


if __name__ == "__main__":
    main()
