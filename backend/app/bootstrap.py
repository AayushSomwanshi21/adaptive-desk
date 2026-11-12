"""Train and promote the first model, so the API has something to serve.

    python -m backend.app.bootstrap                 # all categories in the data
    python -m backend.app.bootstrap --classes 70    # hold some categories back

Held-back categories are written to a CSV of "demo tickets" that
`backend.app.demo_feed` can send to the running API, so you can watch a new
category go through review, approval and training.
"""

import argparse
from pathlib import Path

import numpy as np

from backend.app.config import ROOT, settings
from ml.continual_learning.stage_update import UpdateSettings, train_initial_version
from ml.datasets.loader import load_dataset
from ml.datasets.validator import validate_dataset
from ml.serving.versioning import ModelRegistry

TRAIN_PATH = ROOT / "ml/datasets/train.csv"
DEFAULT_DEMO_PATH = ROOT / "backend/data/demo_tickets.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--classes", type=int, default=None,
                        help="train on this many categories (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--memory-size", type=int, default=1000)
    parser.add_argument("--demo-output", type=Path, default=DEFAULT_DEMO_PATH)
    args = parser.parse_args()

    registry = ModelRegistry(settings.registry_root)
    if registry.active() is not None:
        raise SystemExit(
            f"A model is already active ({registry.active()}). Delete "
            f"{settings.registry_root} to start over.")

    train_df = validate_dataset(
        load_dataset(str(TRAIN_PATH)), remove_duplicates=True)
    train_df["label"] = train_df["label"].astype(str)

    classes = sorted(train_df["label"].unique())
    np.random.default_rng(args.seed).shuffle(classes)
    initial = classes[:args.classes] if args.classes else classes
    held_back = classes[len(initial):]

    update_settings = UpdateSettings(
        seed=args.seed,
        memory_size=args.memory_size,
        holdout_fraction=settings.holdout_fraction,
        model_name=settings.model_name,
    )
    if args.epochs is not None:
        update_settings.train.epochs = args.epochs

    tickets = train_df[train_df["label"].isin(initial)][["text", "label"]]
    report = train_initial_version(registry, tickets, update_settings)
    registry.promote(report["version"])

    print(f"\nPromoted {report['version']} with {len(initial)} categories.")

    if held_back:
        demo = train_df[train_df["label"].isin(held_back)][["text", "label"]]
        args.demo_output.parent.mkdir(parents=True, exist_ok=True)
        demo.to_csv(args.demo_output, index=False)
        print(f"{len(held_back)} categories were held back: {sorted(held_back)}")
        print(f"Their tickets are in {args.demo_output} for demo_feed.")

    print("Start the API with: uvicorn backend.app.main:app --reload")


if __name__ == "__main__":
    main()
