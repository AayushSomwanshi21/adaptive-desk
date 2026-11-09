import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from transformers import AutoTokenizer

from ml.continual_learning.replay_memory import ClassBalancedMemory
from ml.datasets.incremental_splitter import create_incremental_splits
from ml.datasets.loader import load_dataset
from ml.datasets.validator import validate_dataset
from ml.evaluation.metrics import calculate_metrics
from ml.models.text_classifier import AdaptiveDeskClassifier


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "ml/datasets/train.csv"
TEST_PATH = ROOT / "ml/datasets/test.csv"
CHECKPOINT_ROOT = ROOT / "ml/models/checkpoints"

SEED = 42
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
MODEL_NAME = "distilbert-base-uncased"

MEMORY_SIZE = 500

# Class-balanced sampling strength for replay training batches.
#   0.0 -> plain shuffling (no balancing)
#   1.0 -> fully class-balanced sampling (every seen class equally likely)
#   0.5 -> compromise: old classes are boosted but new classes still get
#          plenty of training signal. Tune this on your data.
BALANCE_ALPHA = 0.5

LAMBDA_KD = 0.25

KD_TEMPERATURE = 2.0

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


def make_loader(dataframe, label_to_id, tokenizer, shuffle, balance_alpha=0.0):
    """Build a DataLoader. With balance_alpha > 0, sample class-balanced.

    Each example gets weight count(class) ** -alpha, so rare classes (the
    replayed old ones) are drawn more often than frequent ones (the new ones).
    """
    dataset = prepare_dataset(dataframe, tokenizer, label_to_id)

    if shuffle and balance_alpha > 0:
        label_ids = dataframe["label"].map(label_to_id).to_numpy()
        counts = np.bincount(label_ids)
        weights = counts[label_ids].astype(np.float64) ** (-balance_alpha)

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,  # sampler and shuffle are mutually exclusive
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def distillation_loss(student_logits, teacher_logits, temperature):
    """KL(teacher || student) over the OLD classes only.

    The teacher was trained before the new classes existed, so it only has
    `num_old` outputs. We compare those against the first `num_old` student
    outputs (label IDs are preserved across stages, so they line up).
    """
    num_old = teacher_logits.size(1)
    student_old = student_logits[:, :num_old]

    kd = F.kl_div(
        F.log_softmax(student_old / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    )
    # T^2 keeps the gradient scale comparable to the cross-entropy term.
    return kd * (temperature ** 2)


def train_model(model, loader, teacher=None):
    """Train the model. If a frozen teacher is given, add a KD loss to it."""
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_kd = 0.0

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

            # distillation: keep old-class outputs close to the teacher's,
            # on every example in the batch (new-class data and replay).
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                kd_loss = distillation_loss(
                    logits, teacher_logits, KD_TEMPERATURE
                )
                loss = loss + LAMBDA_KD * kd_loss
                total_kd += kd_loss.item()

            # backpropagation and calculate the gradient
            loss.backward()
            # update the weights
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / max(len(loader), 1)
        message = (
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Training loss: {average_loss:.4f}"
        )
        if teacher is not None:
            average_kd = total_kd / max(len(loader), 1)
            message += f" (KD term: {average_kd:.4f})"
        print(message)


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


def save_checkpoint(model, tokenizer, label_to_id, metrics, output_dir,
                    memory=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model_state.pt")
    tokenizer.save_pretrained(output_dir)

    with open(output_dir / "label_to_id.json", "w", encoding="utf-8") as file:
        json.dump(label_to_id, file, indent=2)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # Save the exact exemplars stored after this stage, for inspection.
    if memory is not None:
        memory.as_dataframe().to_csv(output_dir / "memory.csv", index=False)


def train_stage_one(stage, train_df, tokenizer):
    """Train the initial model on the first 10 categories from scratch."""
    new_classes = [str(category) for category in stage["new_classes"]]
    label_to_id = {category: index for index,
                   category in enumerate(new_classes)}

    model = AdaptiveDeskClassifier(num_labels=len(label_to_id))
    model.to(DEVICE)

    stage_train = stage["train"].copy()
    stage_test = stage["test"].copy()

    print("\n" + "=" * 60)
    print("Stage 1")
    print(f"New categories: {len(new_classes)}")
    print(f"Training examples: {len(stage_train)}")
    print(f"Test examples: {len(stage_test)}")

    train_model(model, make_loader(
        stage_train, label_to_id, tokenizer, shuffle=True))
    metrics = evaluate_model(
        model, stage_test, label_to_id, tokenizer, old_classes=[], new_classes=new_classes
    )
    metrics.update({
        "stage": 1,
        "strategy": "baseline",
        "training_examples": int(len(stage_train)),
        "replay_examples": 0,
    })

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

    output_dir = CHECKPOINT_ROOT / "stage1_baseline"
    save_checkpoint(model, tokenizer, label_to_id, metrics, output_dir)
    print(f"Checkpoint saved to: {output_dir}")
    return model, label_to_id, metrics


def load_checkpoint(checkpoint_dir):
    with open(checkpoint_dir / "label_to_id.json", "r", encoding="utf-8") as file:
        label_to_id = json.load(file)

    model = AdaptiveDeskClassifier(num_labels=len(label_to_id))
    model.load_state_dict(
        torch.load(checkpoint_dir / "model_state.pt",
                   map_location="cpu", weights_only=True)
    )
    model.to(DEVICE)
    return model, label_to_id


def run_experiment(strategy, stages, train_df, tokenizer):
    """Run stages 1-8 for one strategy.

    baseline  : fine-tune on each stage's new data only
    replay    : + class-balanced replay memory
    replay_kd : + class-balanced replay memory and knowledge distillation
    """
    if strategy not in {"baseline", "replay", "replay_kd"}:
        raise ValueError(
            "strategy must be 'baseline', 'replay' or 'replay_kd'")

    uses_replay = strategy in {"replay", "replay_kd"}
    uses_kd = strategy == "replay_kd"

    set_seed()
    print(f"\nUsing device: {DEVICE}")
    print(f"Strategy: {strategy}")

    results = []
    memory = None

    if strategy == "baseline":
        model, label_to_id, stage1_metrics = train_stage_one(
            stages[0], train_df, tokenizer
        )
        results.append(stage1_metrics)
        previous_dir = CHECKPOINT_ROOT / "stage1_baseline"
    else:
        # Stage 1 is shared: replay starts from the same initial model as baseline.
        previous_dir = CHECKPOINT_ROOT / "stage1_baseline"
        if not previous_dir.exists():
            raise FileNotFoundError(
                f"Missing Stage 1 baseline checkpoint: {previous_dir}. "
                "Run the baseline experiment first."
            )
        model, label_to_id = load_checkpoint(previous_dir)
        with open(previous_dir / "metrics.json", "r", encoding="utf-8") as file:
            stage1_metrics = json.load(file)
        stage1_metrics["strategy"] = strategy
        results.append(stage1_metrics)

        # Fill the buffer ONCE from Stage 1 data (the only data available then).
        memory = ClassBalancedMemory(MEMORY_SIZE, seed=SEED)
        memory.update(stages[0]["train"], stages[0]["new_classes"])
        print(f"Replay memory after Stage 1: {memory.summary()}")

    # Train incremental stages 2 through 8.
    for stage_index in range(1, len(stages)):
        stage = stages[stage_index]
        stage_number = stage_index + 1

        old_classes = [str(c) for c in stages[stage_index - 1]["seen_classes"]]
        new_classes = [str(c) for c in stage["new_classes"]]
        seen_classes = [str(c) for c in stage["seen_classes"]]

        print("\n" + "=" * 60)
        print(f"Stage {stage_number}")
        print(f"Previously seen categories: {len(old_classes)}")
        print(f"New categories: {len(new_classes)}")
        print(f"Total categories: {len(seen_classes)}")

        stage_new_data = stage["train"].copy()  # current-stage data only
        stage_train = stage_new_data
        stage_test = stage["test"].copy()

        # Append new label IDs while preserving every previously assigned ID.
        for category in new_classes:
            if category not in label_to_id:
                label_to_id[category] = len(label_to_id)

        if set(label_to_id) != set(seen_classes):
            raise ValueError(
                f"Label mapping does not match Stage {stage_number} classes.")
        if len(label_to_id) != len(seen_classes):
            raise ValueError("Duplicate or missing label IDs detected.")

        # Teacher = frozen copy of the model from the end of the previous
        # stage. It must be copied BEFORE the classifier head is expanded,
        # so its outputs cover exactly the old classes.
        teacher = None
        if uses_kd:
            teacher = copy.deepcopy(model)
            teacher.to(DEVICE)
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad = False

        model.expand_classifier(len(label_to_id))
        model.to(DEVICE)

        replay_count = 0
        balance_alpha = 0.0
        if uses_replay:
            # Replay ONLY from the persistent buffer, never from train_df.
            memory_df = memory.as_dataframe()
            replay_count = len(memory_df)
            stage_train = pd.concat(
                [stage_new_data, memory_df], ignore_index=True)
            balance_alpha = BALANCE_ALPHA
            print(f"Replay examples: {replay_count} ({memory.summary()})")

        print(f"Training examples: {len(stage_train)}")
        print(f"Test examples: {len(stage_test)}")

        train_model(
            model,
            make_loader(
                stage_train, label_to_id, tokenizer, shuffle=True,
                balance_alpha=balance_alpha),
            teacher=teacher,
        )
        del teacher  # free memory before the next stage
        metrics = evaluate_model(
            model, stage_test, label_to_id, tokenizer, old_classes, new_classes
        )
        metrics.update({
            "stage": stage_number,
            "strategy": strategy,
            "training_examples": int(len(stage_train)),
            "replay_examples": int(replay_count),
        })

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"Old-category accuracy: {metrics['old_category_accuracy']:.4f}")
        print(f"New-category accuracy: {metrics['new_category_accuracy']:.4f}")

        # After training, store exemplars of the NEW classes (from this stage's
        # data only) and shrink every class to the new equal quota.
        if uses_replay:
            memory.update(stage_new_data, new_classes)
            print(
                f"Replay memory after Stage {stage_number}: {memory.summary()}")

        output_dir = CHECKPOINT_ROOT / f"stage{stage_number}_{strategy}"
        save_checkpoint(model, tokenizer, label_to_id, metrics, output_dir,
                        memory=memory)
        results.append(metrics)
        previous_dir = output_dir
        print(f"Checkpoint saved to: {output_dir}")

    results_dir = ROOT / "ml" / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"results_{strategy}_all_stages.csv"
    pd.DataFrame(results).sort_values(
        "stage").to_csv(results_path, index=False)

    print(f"\nAll stages completed for {strategy}.")
    print(f"Results saved to: {results_path}")


def main():
    train_df, test_df = load_data()
    stages = create_incremental_splits(
        train_df, test_df, classes_per_stage=10, seed=SEED
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Run baseline first to create the shared Stage 1 checkpoint.
    run_experiment("baseline", stages, train_df, tokenizer)
    # # Replay then starts from that same Stage 1 checkpoint.
    # run_experiment("replay", stages, train_df, tokenizer)
    # # Same replay setup plus knowledge distillation, for a direct comparison.
    run_experiment("replay_kd", stages, train_df, tokenizer)


if __name__ == "__main__":
    main()
