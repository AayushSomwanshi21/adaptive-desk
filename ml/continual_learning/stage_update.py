"""Live stage update: confirmed tickets in, a new (unpromoted) version out.

Two entry points:

    train_initial_version(registry, tickets)
        Train the very first model on a starting set of categories.

    run_stage_update(registry, tickets)
        Take the active version, add the categories found in `tickets`, train
        one incremental stage (class-balanced replay + balanced sampling +
        knowledge distillation) and write a new version folder.

Neither promotes anything. The returned report contains the promotion gate
result; promoting stays a deliberate `registry.promote(name)`.

Held-out data: a share of the tickets is set aside and NEVER trained on. It is
appended to the version's `eval_set.jsonl`, which grows with every version, and
is used to (1) judge the new model against the old one on the same tickets and
(2) fit the temperature / review threshold. The caller therefore should not
feed those tickets back as training data.
"""

import copy
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from ml.continual_learning import training
from ml.continual_learning.experiment_utils import make_validation_split
from ml.continual_learning.promotion import GateSettings, evaluate_gate
from ml.continual_learning.replay_memory import ClassBalancedMemory
from ml.models.text_classifier import AdaptiveDeskClassifier
from ml.serving.predictor import Predictor
from ml.serving.versioning import (
    EVAL_SET_FILE,
    LABELS_FILE,
    MEMORY_DIR,
    METRICS_FILE,
    MODEL_FILE,
    read_json,
    write_json_atomic,
)


@dataclass
class UpdateSettings:
    # Share of the incoming tickets set aside as held-out data.
    holdout_fraction: float = 0.25

    # Replay buffer capacity; only used when creating the first version
    # (later versions keep the capacity stored with their memory).
    memory_size: int = 1000

    # Settings that came out best in the offline experiments.
    balance_alpha: float = 0.5
    lambda_kd: float = 0.25

    # A category needs at least this many tickets before it can be trained on.
    min_examples_per_category: int = 8

    # Calibration: precision target for auto-labelling, and the minimum number
    # of auto-labelled held-out tickets needed to trust a threshold.
    target_precision: float = 0.95
    min_accepted_for_threshold: int = 20

    seed: int = 42
    model_name: str = "distilbert-base-uncased"

    gate: GateSettings = field(default_factory=GateSettings)
    train: training.TrainConfig = field(default_factory=training.TrainConfig)


# ---- helpers ---------------------------------------------------------------
def _clean_tickets(tickets):
    missing = {"text", "label"} - set(tickets.columns)
    if missing:
        raise ValueError(f"tickets is missing columns: {sorted(missing)}")

    frame = tickets.copy()
    frame["text"] = frame["text"].astype(str)
    frame["label"] = frame["label"].astype(str)
    frame = frame[frame["text"].str.strip() != ""]
    if frame.empty:
        raise ValueError("No usable tickets were given.")
    return frame.reset_index(drop=True)


def _check_enough_examples(frame, categories, settings):
    counts = frame["label"].value_counts()
    too_few = {c: int(counts.get(c, 0)) for c in categories
               if counts.get(c, 0) < settings.min_examples_per_category}
    if too_few:
        raise ValueError(
            f"These categories have fewer than "
            f"{settings.min_examples_per_category} tickets and cannot be "
            f"trained yet: {too_few}")


def _read_eval_set(version_dir):
    path = version_dir / EVAL_SET_FILE
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame({"text": [], "label": []})

    frame = pd.read_json(path, lines=True, dtype=False, convert_dates=False)
    frame["label"] = frame["label"].astype(str)
    return frame


def _write_eval_set(path, frame):
    if frame.empty:
        path.write_text("", encoding="utf-8")
    else:
        frame.to_json(path, orient="records", lines=True, force_ascii=False)


def _load_model(version_dir, device):
    label_to_id = read_json(version_dir / LABELS_FILE)
    model = AdaptiveDeskClassifier(num_labels=len(label_to_id))
    model.load_state_dict(torch.load(
        version_dir / MODEL_FILE, map_location="cpu", weights_only=True))
    return model.to(device), label_to_id


def _load_tokenizer(version_dir, model_name):
    if (version_dir / "tokenizer_config.json").exists():
        return AutoTokenizer.from_pretrained(version_dir)
    return AutoTokenizer.from_pretrained(model_name)


def _accuracy(y_true, y_pred, mask=None):
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return None
    return float((y_true == y_pred).mean())


def _per_category(y_true, y_pred, id_to_label):
    rows = []
    for category_id in sorted(set(y_true.tolist())):
        mask = y_true == category_id
        rows.append({
            "category": id_to_label[category_id],
            "n_eval": int(mask.sum()),
            "accuracy": float((y_pred[mask] == category_id).mean()),
        })
    return rows


def _calibrate(model, tokenizer, label_to_id, eval_set, settings):
    """Fit temperature + review threshold on the held-out pool."""
    if eval_set.empty:
        return {"temperature": 1.0, "threshold": None,
                "note": "No held-out data: every ticket will go to review."}

    predictor = Predictor(
        model, tokenizer, label_to_id,
        device=settings.train.device, max_length=settings.train.max_length)
    return predictor.calibrate(
        eval_set["text"].tolist(), eval_set["label"].tolist(),
        target_precision=settings.target_precision,
        min_accepted=settings.min_accepted_for_threshold)


def _settings_summary(settings):
    return {
        "holdout_fraction": settings.holdout_fraction,
        "balance_alpha": settings.balance_alpha,
        "lambda_kd": settings.lambda_kd,
        "epochs": settings.train.epochs,
        "learning_rate": settings.train.learning_rate,
        "batch_size": settings.train.batch_size,
        "target_precision": settings.target_precision,
        "seed": settings.seed,
    }


def _write_version(registry, parent, notes, model, tokenizer, label_to_id,
                   memory, eval_set, calibration, report):
    """Create the version folder and write everything into it.

    The weights and the label map are written LAST, because their presence is
    what makes a version "complete" (promotable). If anything fails, the
    half-written folder is removed.
    """
    name, path = registry.create_version(parent=parent, notes=notes)
    try:
        report["version"] = name
        memory.save(path / MEMORY_DIR)
        _write_eval_set(path / EVAL_SET_FILE, eval_set)
        write_json_atomic(path / METRICS_FILE, report)
        registry.write_calibration(name, calibration)
        tokenizer.save_pretrained(path)

        torch.save(model.state_dict(), path / MODEL_FILE)
        write_json_atomic(path / LABELS_FILE, label_to_id)
    except Exception:
        shutil.rmtree(path, ignore_errors=True)
        raise
    return name


# ---- first version ---------------------------------------------------------
def train_initial_version(registry, tickets, settings=None):
    """Train the first model from scratch on a starting set of categories."""
    settings = settings or UpdateSettings()
    config = settings.train
    training.set_seed(settings.seed)

    tickets = _clean_tickets(tickets)
    categories = sorted(set(tickets["label"]))
    _check_enough_examples(tickets, categories, settings)

    train_part, heldout = make_validation_split(
        tickets, settings.holdout_fraction, settings.seed)

    label_to_id = {category: index for index,
                   category in enumerate(categories)}
    model = AdaptiveDeskClassifier(
        num_labels=len(label_to_id)).to(config.device)
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)

    print(f"Initial version: {len(categories)} categories, "
          f"{len(train_part)} training tickets, {len(heldout)} held out")

    loader = training.make_loader(
        train_part[["text", "label"]], label_to_id, tokenizer,
        shuffle=True, config=config)
    training.train_model(model, loader, config)

    memory = ClassBalancedMemory(settings.memory_size, seed=settings.seed)
    memory.update(train_part, categories)

    y_true, y_pred = training.collect_predictions(
        model, heldout, label_to_id, tokenizer, config)
    id_to_label = {i: label for label, i in label_to_id.items()}

    calibration = _calibrate(model, tokenizer, label_to_id, heldout, settings)

    report = {
        "kind": "initial",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent": None,
        "categories": categories,
        "new_categories": categories,
        "n_training_tickets": int(len(train_part)),
        "n_replay_examples": 0,
        "n_heldout_new": int(len(heldout)),
        "eval_pool_size": int(len(heldout)),
        "accuracy": _accuracy(y_true, y_pred),
        "per_category": _per_category(y_true, y_pred, id_to_label),
        "calibration": calibration,
        "memory": memory.summary(),
        "gate": None,
        "settings": _settings_summary(settings),
    }

    name = _write_version(
        registry, None, f"initial model, {len(categories)} categories",
        model, tokenizer, label_to_id, memory, heldout, calibration, report)
    print(f"Created {name} (accuracy on held-out: {report['accuracy']:.3f})")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


# ---- incremental update ----------------------------------------------------
def run_stage_update(registry, tickets, settings=None, base_version=None):
    """Train one incremental stage on top of `base_version` (default: active).

    `tickets` is a DataFrame with "text" and "label" of newly CONFIRMED
    tickets: both tickets for brand-new categories and extra tickets for
    existing ones. Returns the report (including the promotion gate result)
    and leaves promotion to the caller.
    """
    settings = settings or UpdateSettings()
    config = settings.train
    training.set_seed(settings.seed)

    base_version = base_version or registry.active()
    if base_version is None:
        raise ValueError(
            "There is no base version yet; run train_initial_version first.")
    base_dir = registry.version_dir(base_version)

    tickets = _clean_tickets(tickets)
    model, label_to_id = _load_model(base_dir, config.device)
    tokenizer = _load_tokenizer(base_dir, settings.model_name)
    memory = ClassBalancedMemory.load(base_dir / MEMORY_DIR)

    base_label_to_id = dict(label_to_id)
    new_categories = sorted(set(tickets["label"]) - set(label_to_id))
    _check_enough_examples(tickets, new_categories, settings)

    # ---- hold out data that is never trained on ---------------------------
    train_part, heldout = make_validation_split(
        tickets, settings.holdout_fraction, settings.seed)

    # check if there are enough samples collected for training.
    needed = settings.gate.min_new_eval_examples
    held_counts = heldout["label"].value_counts()
    short = {c: int(held_counts.get(c, 0)) for c in new_categories
             if held_counts.get(c, 0) < needed}
    if short:
        raise ValueError(
            f"Still collecting examples: these new categories have fewer than "
            f"{needed} held-out tickets (about "
            f"{int(needed / settings.holdout_fraction)} tickets per category "
            f"are needed with the current hold-out): {short}")

    prior_eval = _read_eval_set(base_dir)
    # Guard against leaks: drop training tickets whose text is already in the
    # permanent evaluation set (e.g. the same ticket submitted twice).
    leaked = train_part["text"].isin(set(prior_eval["text"]))
    train_part = train_part[~leaked].reset_index(drop=True)

    eval_pool = pd.concat([prior_eval, heldout], ignore_index=True)

    print(f"Stage update from {base_version}: "
          f"{len(new_categories)} new categories {new_categories[:5]}, "
          f"{len(train_part)} training tickets, {len(heldout)} held out, "
          f"{int(leaked.sum())} dropped as duplicates of evaluation tickets")

    # ---- the frozen teacher = the base model ------------------------------
    teacher = copy.deepcopy(model)
    teacher.to(config.device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    num_old = len(label_to_id)
    old_rows = eval_pool[eval_pool["label"].isin(base_label_to_id)]
    if len(old_rows):
        base_true, base_pred = training.collect_predictions(
            teacher, old_rows, base_label_to_id, tokenizer, config)
        base_old_accuracy = _accuracy(base_true, base_pred)
    else:
        base_old_accuracy = None

    # ---- expand the head and train ---------------------------------------
    for category in new_categories:
        label_to_id[category] = len(label_to_id)
    if new_categories:
        model.expand_classifier(len(label_to_id))
    model.to(config.device)

    memory_df = memory.as_dataframe()
    train_pool = pd.concat(
        [train_part[["text", "label"]], memory_df[["text", "label"]]],
        ignore_index=True)
    print(f"Training on {len(train_part)} new tickets + "
          f"{len(memory_df)} replayed examples ({memory.summary()})")

    loader = training.make_loader(
        train_pool, label_to_id, tokenizer, shuffle=True, config=config,
        balance_alpha=settings.balance_alpha)
    training.train_model(
        model, loader, config, teacher=teacher, num_old=num_old,
        lambda_kd=settings.lambda_kd)
    del teacher

    # Store exemplars of the NEW categories only, then rebalance every class.
    memory.update(train_part, new_categories)

    # ---- evaluate on the pooled held-out data -----------------------------
    id_to_label = {i: label for label, i in label_to_id.items()}
    old_ids = [base_label_to_id[c] for c in base_label_to_id]
    new_ids = [label_to_id[c] for c in new_categories]

    y_true, y_pred = training.collect_predictions(
        model, eval_pool, label_to_id, tokenizer, config)
    old_mask = np.isin(y_true, old_ids)
    new_mask = np.isin(y_true, new_ids)

    new_old_accuracy = _accuracy(y_true, y_pred, old_mask)
    new_category_accuracy = _accuracy(y_true, y_pred, new_mask)

    new_set = set(new_categories)
    new_results = [row for row in _per_category(y_true, y_pred, id_to_label)
                   if row["category"] in new_set]

    gate = evaluate_gate(
        base_old_accuracy, new_old_accuracy, int(old_mask.sum()),
        new_results, settings.gate)

    calibration = _calibrate(
        model, tokenizer, label_to_id, eval_pool, settings)

    report = {
        "kind": "stage_update",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent": base_version,
        "categories": list(label_to_id),
        "new_categories": new_categories,
        "n_training_tickets": int(len(train_part)),
        "n_replay_examples": int(len(memory_df)),
        "n_heldout_new": int(len(heldout)),
        "n_dropped_duplicates": int(leaked.sum()),
        "eval_pool_size": int(len(eval_pool)),
        "base_old_accuracy": base_old_accuracy,
        "old_accuracy": new_old_accuracy,
        "old_accuracy_drop": (
            None if base_old_accuracy is None or new_old_accuracy is None
            else base_old_accuracy - new_old_accuracy),
        "new_category_accuracy": new_category_accuracy,
        "accuracy": _accuracy(y_true, y_pred),
        "per_category": _per_category(y_true, y_pred, id_to_label),
        "calibration": calibration,
        "memory": memory.summary(),
        "gate": gate,
        "settings": _settings_summary(settings),
    }

    name = _write_version(
        registry, base_version,
        f"stage update from {base_version}: +{len(new_categories)} categories",
        model, tokenizer, label_to_id, memory, eval_pool, calibration, report)

    verdict = "PASSED" if gate["passed"] else "FAILED"
    print(f"Created {name}; promotion gate {verdict}")
    for check in gate["checks"]:
        print(f"  [{'ok' if check['passed'] else 'x'}] {check['detail']}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report
