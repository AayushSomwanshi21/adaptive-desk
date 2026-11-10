"""Simulate the live ticket loop on your dataset.

    1. Train an initial model on the first N categories (default 70).
    2. The remaining categories "arrive" a few at a time. For each group,
       `--tickets-per-class` confirmed tickets per new category are collected
       and one stage update is run.
    3. If the promotion gate passes, the new version is promoted; if not, the
       tickets stay pending and are added to the next update.
    4. Every candidate version is scored on the REAL test set (restricted to
       the categories it knows), which the update itself never sees.

Examples (run from the repo root):

    python -m ml.experiments.simulate_live_loop
    python -m ml.experiments.simulate_live_loop --new-per-update 1 --tickets-per-class 25
    python -m ml.experiments.simulate_live_loop --new-per-update 2 --tickets-per-class 10

Compare --new-per-update 1, 2 and 10, and --tickets-per-class 10, 25 and 50:
the first shows how well the update works when categories arrive one at a time
(your offline experiments added 10 at once); the second shows how many tickets
are worth waiting for before retraining.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.continual_learning.stage_update import (
    UpdateSettings,
    run_stage_update,
    train_initial_version,
)
from ml.continual_learning.training import TrainConfig
from ml.datasets.loader import load_dataset
from ml.datasets.validator import validate_dataset
from ml.serving.predictor import Predictor
from ml.serving.versioning import ModelRegistry

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "ml/datasets/train.csv"
TEST_PATH = ROOT / "ml/datasets/test.csv"
SIMULATION_ROOT = ROOT / "ml/experiments/simulation"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--initial-classes", type=int, default=70)
    parser.add_argument("--new-per-update", type=int, default=1,
                        help="categories that arrive together in one update")
    parser.add_argument("--tickets-per-class", type=int, default=25,
                        help="confirmed tickets collected per new category "
                             "before retraining")
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--memory-size", type=int, default=1000)
    parser.add_argument("--balance-alpha", type=float, default=0.5)
    parser.add_argument("--lambda-kd", type=float, default=0.25)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="delete an existing simulation with this tag")
    return parser.parse_args()


def load_frames():
    train_df = validate_dataset(
        load_dataset(str(TRAIN_PATH)), remove_duplicates=True)
    test_df = validate_dataset(
        load_dataset(str(TEST_PATH)), remove_duplicates=False)
    for frame in (train_df, test_df):
        frame["label"] = frame["label"].astype(str)
    return train_df, test_df


def evaluate_on_test(registry, version, test_df, base_classes, new_classes):
    """Accuracy of a version on the real test set, for the classes it knows."""
    predictor = Predictor.from_version(registry.version_dir(version))

    known = test_df[test_df["label"].isin(predictor.label_to_id)]
    logits = predictor.logits(known["text"].tolist())
    predicted = logits.argmax(axis=1)
    truth = known["label"].map(predictor.label_to_id).to_numpy()
    correct = predicted == truth

    labels = known["label"].to_numpy()
    old_mask = np.isin(labels, list(base_classes))
    new_mask = np.isin(labels, list(new_classes))

    result = {
        "test_accuracy": float(correct.mean()),
        "test_old_accuracy": (
            float(correct[old_mask].mean()) if old_mask.any() else None),
        "test_new_accuracy": (
            float(correct[new_mask].mean()) if new_mask.any() else None),
        "n_known_categories": len(predictor.label_to_id),
    }

    del predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    tag = args.tag or (
        f"init{args.initial_classes}_new{args.new_per_update}"
        f"_t{args.tickets_per_class}_s{args.seed}")
    run_dir = SIMULATION_ROOT / tag
    if run_dir.exists():
        if not args.overwrite:
            raise SystemExit(
                f"{run_dir} already exists; pass --overwrite to replace it.")
        shutil.rmtree(run_dir)
    registry = ModelRegistry(run_dir / "registry")

    train_df, test_df = load_frames()

    classes = sorted(train_df["label"].unique())
    rng.shuffle(classes)
    initial_classes = classes[:args.initial_classes]
    arrivals = classes[args.initial_classes:]
    if not arrivals:
        raise SystemExit("--initial-classes leaves no categories to arrive.")

    train_config = TrainConfig()
    if args.epochs is not None:
        train_config.epochs = args.epochs
    settings = UpdateSettings(
        holdout_fraction=args.holdout_fraction,
        memory_size=args.memory_size,
        balance_alpha=args.balance_alpha,
        lambda_kd=args.lambda_kd,
        seed=args.seed,
        train=train_config,
    )

    rows = []

    # ---- initial version ---------------------------------------------------
    print(f"\n{'=' * 60}\nInitial model on {len(initial_classes)} categories")
    initial_tickets = train_df[train_df["label"].isin(initial_classes)]
    train_initial_version(registry, initial_tickets, settings)
    registry.promote("v0001")

    served = evaluate_on_test(
        registry, "v0001", test_df, base_classes=[], new_classes=initial_classes)
    served_accuracy = served["test_accuracy"]
    rows.append({
        "update": 0, "new_version": "v0001", "base_version": None,
        "new_categories": len(initial_classes), "tickets_used": len(initial_tickets),
        "gate_passed": True, "promoted": True,
        "served_accuracy": served_accuracy, **served,
    })

    # ---- arriving categories ----------------------------------------------
    groups = [arrivals[i:i + args.new_per_update]
              for i in range(0, len(arrivals), args.new_per_update)]
    if args.max_updates is not None:
        groups = groups[:args.max_updates]

    pending = pd.DataFrame(columns=["text", "label"])
    known_classes = list(initial_classes)

    for number, group in enumerate(groups, start=1):
        print(f"\n{'=' * 60}\nUpdate {number}/{len(groups)}: "
              f"new categories {group}")

        collected = []
        for category in group:
            pool = train_df[train_df["label"] == category]
            take = min(args.tickets_per_class, len(pool))
            collected.append(pool.sample(
                n=take, random_state=int(rng.integers(0, 2**31 - 1))))
        tickets = pd.concat([pending] + collected, ignore_index=True)

        base_version = registry.active()
        report = run_stage_update(registry, tickets, settings)
        new_version = report["version"]
        passed = report["gate"]["passed"]

        scores = evaluate_on_test(
            registry, new_version, test_df,
            base_classes=known_classes,
            new_classes=report["new_categories"])

        if passed:
            registry.promote(new_version)
            known_classes = known_classes + list(group)
            served_accuracy = scores["test_accuracy"]
            pending = pending.iloc[0:0]
        else:
            # Keep the tickets and retry together with the next arrivals.
            pending = tickets[["text", "label"]]

        rows.append({
            "update": number, "new_version": new_version,
            "base_version": base_version,
            "new_categories": len(group), "tickets_used": len(tickets),
            "gate_passed": passed, "promoted": passed,
            "gate_old_drop": report["old_accuracy_drop"],
            "held_out_new_accuracy": report["new_category_accuracy"],
            "served_accuracy": served_accuracy, **scores,
        })

        pd.DataFrame(rows).to_csv(
            run_dir / "simulation_results.csv", index=False)

    # ---- summary -----------------------------------------------------------
    results = pd.DataFrame(rows)
    results.to_csv(run_dir / "simulation_results.csv", index=False)

    shown = results[[
        "update", "new_version", "n_known_categories", "gate_passed",
        "test_accuracy", "test_old_accuracy", "test_new_accuracy",
        "served_accuracy"]].copy()
    for column in shown.columns[4:]:
        shown[column] = (100 * shown[column]).round(1)

    print(f"\n{'=' * 60}\nSimulation '{tag}' (test accuracy in %)")
    print(shown.to_string(index=False))
    print(f"\nSaved to {run_dir / 'simulation_results.csv'}")
    print(f"Versions in {registry.root}; active: {registry.active()}")


if __name__ == "__main__":
    main()
