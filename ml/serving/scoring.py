"""Numpy-only scoring helpers for serving.

Everything that turns raw model logits into decisions lives here, with no
torch import, so it can be unit-tested and reused by the API and by scripts.

Pipeline: logits -> temperature scaling (calibration) -> top-k predictions ->
"needs review" decision using a confidence threshold chosen on held-out data.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


def softmax(logits, temperature=1.0):
    """Row-wise softmax of logits / temperature (numerically stable)."""
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def negative_log_likelihood(logits, labels, temperature=1.0):
    """Mean NLL of the true labels, computed with a stable log-softmax.

    No clipping of probabilities: clipping flattens the loss for very small
    temperatures and would fool the search in `fit_temperature`.
    """
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-np.mean(log_probs[np.arange(len(labels)), labels]))


def fit_temperature(logits, labels, low=0.05, high=20.0, iterations=80):
    """Find the temperature T that minimises NLL on held-out logits.

    Raw softmax scores are usually overconfident (T > 1 fixes that). The loss
    is convex in beta = 1 / T, so a golden-section search is enough and needs
    no scipy.

    `logits` must come from data the model was NOT trained on.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) == 0:
        raise ValueError("Cannot fit a temperature without examples.")

    golden = (np.sqrt(5.0) - 1.0) / 2.0
    left, right = 1.0 / high, 1.0 / low

    def loss(beta):
        return negative_log_likelihood(logits, labels, 1.0 / beta)

    a = right - golden * (right - left)
    b = left + golden * (right - left)
    loss_a, loss_b = loss(a), loss(b)

    for _ in range(iterations):
        if loss_a < loss_b:
            right, b, loss_b = b, a, loss_a
            a = right - golden * (right - left)
            loss_a = loss(a)
        else:
            left, a, loss_a = a, b, loss_b
            b = left + golden * (right - left)
            loss_b = loss(b)

    return float(1.0 / ((left + right) / 2.0))


def expected_calibration_error(confidences, correct, bins=10):
    """Average gap between stated confidence and actual accuracy (0 is best)."""
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)

    bin_ids = np.minimum((confidences * bins).astype(int), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = bin_ids == index
        if mask.any():
            gap = abs(correct[mask].mean() - confidences[mask].mean())
            error += mask.mean() * gap
    return float(error)


def choose_threshold(confidences, correct, target_precision=0.95,
                     min_accepted=20):
    """Pick the review threshold from a precision target.

    Tickets with confidence >= threshold are auto-labelled. This returns the
    LOWEST threshold (the most auto-labelled tickets) whose auto-labelled
    tickets are still at least `target_precision` correct, requiring at least
    `min_accepted` of them so a lucky handful at the top cannot decide it.

    Returns a dict. If no threshold reaches the target, "threshold" is None,
    which the predictor treats as "send everything to review".
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    total = len(confidences)

    empty = {"threshold": None, "precision": None, "auto_rate": 0.0,
             "review_rate": 1.0, "accepted": 0}
    if total == 0:
        return empty

    order = np.argsort(-confidences, kind="stable")
    sorted_conf = confidences[order]
    counts = np.arange(1, total + 1)
    precision = np.cumsum(correct[order]) / counts

    # Cut only after the last of a group of tied confidences.
    is_group_end = np.append(sorted_conf[1:] != sorted_conf[:-1], True)
    valid = is_group_end & (counts >= min_accepted) & (
        precision >= target_precision)

    if not valid.any():
        return empty

    best = int(np.flatnonzero(valid).max())
    return {
        "threshold": float(sorted_conf[best]),
        "precision": float(precision[best]),
        "auto_rate": float(counts[best] / total),
        "review_rate": float(1.0 - counts[best] / total),
        "accepted": int(counts[best]),
    }


def threshold_curve(confidences, correct, points=40):
    """Precision / auto-label rate for a sweep of thresholds (for the UI)."""
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)

    thresholds = np.unique(
        np.quantile(confidences, np.linspace(0.0, 1.0, points)))
    rows = []
    for threshold in thresholds:
        accepted = confidences >= threshold
        rows.append({
            "threshold": float(threshold),
            "auto_rate": float(accepted.mean()),
            "review_rate": float(1.0 - accepted.mean()),
            "precision": (float(correct[accepted].mean())
                          if accepted.any() else None),
        })
    return pd.DataFrame(rows)


@dataclass
class Prediction:
    label: str                 # top-1 category
    confidence: float          # calibrated probability of the top-1 category
    top: list                  # [(category, probability), ...] best first
    needs_review: bool         # True -> goes to the human review queue


def build_predictions(logits, id_to_label, temperature, threshold, top_k=3):
    """Turn a batch of logits into Prediction objects.

    threshold=None means "no usable threshold": every ticket needs review.
    """
    probs = softmax(logits, temperature)
    k = min(top_k, probs.shape[1])
    top_ids = np.argsort(-probs, axis=1)[:, :k]

    predictions = []
    for row, ids in zip(probs, top_ids):
        top = [(id_to_label[int(i)], float(row[i])) for i in ids]
        confidence = top[0][1]
        predictions.append(Prediction(
            label=top[0][0],
            confidence=confidence,
            top=top,
            needs_review=threshold is None or confidence < threshold,
        ))
    return predictions
