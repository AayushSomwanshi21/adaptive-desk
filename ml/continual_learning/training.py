"""Shared training and evaluation code.

One implementation used by both the experiment runner (continual_runner.py)
and the live stage update (stage_update.py), so the system that serves tickets
trains exactly like the one that was evaluated in the experiments.

Nothing here reads module-level settings: everything that used to be a global
(batch size, epochs, ...) comes in through a TrainConfig.
"""

import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ml.evaluation.metrics import calculate_metrics


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 2e-5
    max_length: int = 128
    kd_temperature: float = 2.0
    ace_mask: str = "stage"          # "stage" or "batch" (see ace_loss)
    device: torch.device = field(default_factory=default_device)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataset(dataframe, tokenizer, label_to_id, config):
    labels = dataframe["label"].map(label_to_id)

    if labels.isna().any():
        raise ValueError(
            "Dataset contains a label missing from the label mapping."
        )

    encodings = tokenizer(
        dataframe["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=config.max_length,
        return_tensors="pt",
    )

    return TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        torch.tensor(labels.to_numpy(), dtype=torch.long),
    )


def make_loader(dataframe, label_to_id, tokenizer, shuffle, config,
                balance_alpha=0.0):
    """Build a DataLoader. With balance_alpha > 0, sample class-balanced.

    Each example gets weight count(class) ** -alpha, so rare classes (the
    replayed old ones) are drawn more often than frequent ones (the new ones).
    """
    dataset = prepare_dataset(dataframe, tokenizer, label_to_id, config)

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
            batch_size=config.batch_size,
            sampler=sampler,  # sampler and shuffle are mutually exclusive
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
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


def ace_loss(logits, labels, num_old, mask_mode="stage"):
    """Asymmetric cross-entropy (ER-ACE).

    - New-class examples (label >= num_old): cross-entropy with the old-class
      logits masked out, so learning a new class never pushes old logits down.
    - Replayed examples (label < num_old): ordinary cross-entropy over all
      classes, which is where old-vs-new discrimination is learned.

    During training the memory only holds old classes, so label < num_old
    identifies replayed examples exactly.
    """
    new_mask = labels >= num_old
    old_mask = ~new_mask
    terms = []

    if new_mask.any():
        new_logits = logits[new_mask]
        new_labels = labels[new_mask]

        if mask_mode == "stage":
            keep = torch.arange(logits.size(
                1), device=logits.device) >= num_old
        elif mask_mode == "batch":
            keep = torch.zeros(
                logits.size(1), dtype=torch.bool, device=logits.device)
            keep[new_labels] = True
        else:
            raise ValueError("ace_mask must be 'stage' or 'batch'")

        new_logits = new_logits.masked_fill(~keep, -1e9)
        terms.append(F.cross_entropy(new_logits, new_labels))

    if old_mask.any():
        terms.append(F.cross_entropy(logits[old_mask], labels[old_mask]))

    return sum(terms)


def train_model(model, loader, config, teacher=None, num_old=0,
                use_ace=False, lambda_kd=1.0):
    """Train the model.

    teacher : frozen previous-stage model; if given, a KD loss is added.
    lambda_kd : weight of the KD term (only used when a teacher is given).
    use_ace : replace plain cross-entropy with ER-ACE's asymmetric version
              (needs num_old > 0, i.e. incremental stages with replay).
    """
    if use_ace and num_old <= 0:
        raise ValueError("ACE needs num_old > 0 (use it from Stage 2 on).")

    device = config.device

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        total_loss = 0.0
        total_kd = 0.0

        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # set gradients to zero
            optimizer.zero_grad()

            # get the logits of the text classifier (DistilBERT)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # calculate the loss
            if use_ace:
                loss = ace_loss(logits, labels, num_old, config.ace_mask)
            else:
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
                    logits, teacher_logits, config.kd_temperature
                )
                loss = loss + lambda_kd * kd_loss
                total_kd += kd_loss.item()

            # backpropagation and calculate the gradient
            loss.backward()
            # update the weights
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / max(len(loader), 1)
        message = (
            f"Epoch {epoch + 1}/{config.epochs} "
            f"- Training loss: {average_loss:.4f}"
        )
        if teacher is not None:
            average_kd = total_kd / max(len(loader), 1)
            message += f" (KD term: {average_kd:.4f})"
        print(message)


def collect_predictions(model, dataframe, label_to_id, tokenizer, config):
    """Run the model over a labelled DataFrame; return (y_true, y_pred) ids."""
    loader = make_loader(
        dataframe, label_to_id, tokenizer, shuffle=False, config=config
    )

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            logits = model(
                input_ids=input_ids.to(config.device),
                attention_mask=attention_mask.to(config.device),
            )
            # selects the index of the max prediction
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    return np.asarray(all_labels), np.asarray(all_predictions)


def evaluate_model(model, test_df, label_to_id, tokenizer,
                   old_classes, new_classes, config):
    y_true, y_pred = collect_predictions(
        model, test_df, label_to_id, tokenizer, config
    )

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
