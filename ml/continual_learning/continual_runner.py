import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from ml.continual_learning.experiment_utils import (
    format_summary,
    make_validation_split,
    paired_differences,
    per_run_metrics,
    summarize_runs,
)
from ml.continual_learning.replay_memory import ClassBalancedMemory
from ml.continual_learning import training
from ml.continual_learning.training import TrainConfig
from ml.datasets.incremental_splitter import create_incremental_splits
from ml.datasets.loader import load_dataset
from ml.datasets.validator import validate_dataset
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

MEMORY_SIZE = 1000

# Class-balanced sampling strength for replay training batches.
#   0.0 -> plain shuffling (no balancing)
#   1.0 -> fully class-balanced sampling (every seen class equally likely)
#   0.5 -> compromise: old classes are boosted but new classes still get
#          plenty of training signal.
BALANCE_ALPHA = 1.0

# --- Knowledge distillation settings (used by the "replay_kd" strategy) ------
# Weight of the distillation loss relative to cross-entropy. Too low and KD
# does nothing; too high and the model resists learning the new classes.
LAMBDA_KD = 1.0

# Softens both probability distributions so the small probabilities on
# non-top classes carry information.
KD_TEMPERATURE = 2.0

# --- ER-ACE settings (used by "replay_ace" and "replay_ace_kd") --------------
# Which logits the new-class examples are allowed to compete against:
#   "stage" -> all classes introduced in the current stage (old-class logits
#              are masked out of the softmax)
#   "batch" -> only the classes present among the new examples of that batch
#              (the original ER-ACE paper's choice; a stronger mask)
ACE_MASK = "stage"

# --- Experiment protocol -----------------------------------------------------
# Seeds to run. Each seed changes the class order (via the splitter), the
# validation split, model init, data order and memory sampling.
SEEDS = [42, 43, 44]

# "val":  train on 80% of the training data and evaluate on the held-out 20%.
#         Use this mode to CHOOSE hyperparameters.
# "test": train on all training data and evaluate on the real test set.
#         Use this mode ONCE, for the final report.
EVAL_MODE = "val"
VAL_FRACTION = 0.2

# Skip an experiment if its results file already exists (lets you resume a
# long sweep). Delete the folder under RESULTS_ROOT to force a rerun, e.g.
# after changing EPOCHS or any other global setting.
RESUME = True

# Saving model weights for every stage of every run needs a lot of disk space
# (~250 MB each). Metrics and memory contents are always saved; the shared
# Stage 1 model is always saved because every experiment starts from it.
SAVE_STAGE_CHECKPOINTS = False

RESULTS_ROOT = ROOT / "ml" / "experiments" / "results"

# Paired differences in the final summary are computed against this one.
REFERENCE_EXPERIMENT = "baseline"

# Each experiment: a unique name, a strategy, and optional hyperparameters.
#   strategy: baseline | replay | replay_kd | replay_ace | replay_ace_kd | joint
#   balance_alpha / lambda_kd fall back to the globals above if omitted.
# "joint" trains from scratch on all data seen up to each stage (an upper
# bound, not a continual-learning method).
EXPERIMENTS = [
    {"name": "baseline", "strategy": "baseline"},
    {"name": "replay_kd_a0.5_l0.25", "strategy": "replay_kd",
     "balance_alpha": 0.5, "lambda_kd": 0.25},
]

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


# The training and evaluation code lives in training.py (shared with the live
# stage update). These thin wrappers keep the runner's call sites unchanged and
# build the TrainConfig from this file's settings, so editing EPOCHS,
# BATCH_SIZE, ... above still works.
def _train_config():
    return TrainConfig(
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        max_length=MAX_LENGTH,
        kd_temperature=KD_TEMPERATURE,
        ace_mask=ACE_MASK,
        device=DEVICE,
    )


def make_loader(dataframe, label_to_id, tokenizer, shuffle, balance_alpha=0.0):
    return training.make_loader(
        dataframe, label_to_id, tokenizer, shuffle, _train_config(),
        balance_alpha=balance_alpha,
    )


def train_model(model, loader, teacher=None, num_old=0, use_ace=False,
                lambda_kd=LAMBDA_KD):
    return training.train_model(
        model, loader, _train_config(), teacher=teacher, num_old=num_old,
        use_ace=use_ace, lambda_kd=lambda_kd,
    )


def evaluate_model(model, test_df, label_to_id, tokenizer,
                   old_classes, new_classes):
    return training.evaluate_model(
        model, test_df, label_to_id, tokenizer, old_classes, new_classes,
        _train_config(),
    )


def save_checkpoint(model, tokenizer, label_to_id, metrics, output_dir,
                    memory=None, save_model=True):
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_model:
        torch.save(model.state_dict(), output_dir / "model_state.pt")
        tokenizer.save_pretrained(output_dir)

    with open(output_dir / "label_to_id.json", "w", encoding="utf-8") as file:
        json.dump(label_to_id, file, indent=2)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # Save the exact exemplars stored after this stage, for inspection.
    if memory is not None:
        memory.as_dataframe().to_csv(output_dir / "memory.csv", index=False)


def train_stage_one(stage, tokenizer, output_dir):
    """Train the initial model on the first 10 categories from scratch."""
    new_classes = [str(category) for category in stage["new_classes"]]
    label_to_id = {category: index for index,
                   category in enumerate(new_classes)}

    model = AdaptiveDeskClassifier(num_labels=len(label_to_id))
    model.to(DEVICE)

    stage_train = stage["train"].copy()
    stage_test = stage["test"].copy()

    print("\n" + "=" * 60)
    print("Stage 1 (shared by every experiment of this seed)")
    print(f"New categories: {len(new_classes)}")
    print(f"Training examples: {len(stage_train)}")
    print(f"Evaluation examples: {len(stage_test)}")

    train_model(model, make_loader(
        stage_train, label_to_id, tokenizer, shuffle=True))
    metrics = evaluate_model(
        model, stage_test, label_to_id, tokenizer,
        old_classes=[], new_classes=new_classes
    )
    metrics.update({
        "stage": 1,
        "strategy": "baseline",
        "training_examples": int(len(stage_train)),
        "replay_examples": 0,
    })

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

    # Always keep the weights: every experiment starts from this model.
    save_checkpoint(model, tokenizer, label_to_id, metrics, output_dir)
    print(f"Stage 1 checkpoint saved to: {output_dir}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def ensure_stage_one(stages, tokenizer, seed, tag):
    """Train Stage 1 once per (mode, seed); later experiments reuse it."""
    stage_one_dir = CHECKPOINT_ROOT / tag / "stage1_shared"

    if (stage_one_dir / "model_state.pt").exists():
        print(f"Reusing Stage 1 checkpoint: {stage_one_dir}")
        return stage_one_dir

    set_seed(seed)
    train_stage_one(stages[0], tokenizer, stage_one_dir)
    return stage_one_dir


def build_stages(train_df, test_df, seed, mode):
    """Create the incremental stages for one seed.

    In "val" mode the real test set is never used: the evaluation data of
    every stage comes from a validation split carved out of the training data.
    """
    if mode == "val":
        fit_df, eval_df = make_validation_split(train_df, VAL_FRACTION, seed)
    elif mode == "test":
        fit_df, eval_df = train_df, test_df
    else:
        raise ValueError("mode must be 'val' or 'test'")

    print(f"Mode: {mode} | train rows: {len(fit_df)} | "
          f"evaluation rows: {len(eval_df)}")
    return create_incremental_splits(
        fit_df, eval_df, classes_per_stage=10, seed=seed
    )


def run_incremental(spec, stages, tokenizer, seed, tag):
    """Stages 2..N for baseline / replay / replay_kd / replay_ace[_kd]."""
    name = spec["name"]
    strategy = spec["strategy"]
    alpha = spec.get("balance_alpha", BALANCE_ALPHA)
    lambda_kd = spec.get("lambda_kd", LAMBDA_KD)

    valid = {"baseline", "replay", "replay_kd", "replay_ace", "replay_ace_kd"}
    if strategy not in valid:
        raise ValueError(f"unknown strategy '{strategy}'")

    uses_replay = strategy != "baseline"
    uses_kd = strategy in {"replay_kd", "replay_ace_kd"}
    uses_ace = strategy in {"replay_ace", "replay_ace_kd"}

    set_seed(seed)
    print(f"\nUsing device: {DEVICE}")
    print(f"Experiment: {name} (strategy={strategy}, "
          f"balance_alpha={alpha if uses_replay else '-'}, "
          f"lambda_kd={lambda_kd if uses_kd else '-'}, seed={seed})")

    # Every experiment starts from the same Stage 1 model for this seed.
    stage_one_dir = CHECKPOINT_ROOT / tag / "stage1_shared"
    model, label_to_id = load_checkpoint(stage_one_dir)
    with open(stage_one_dir / "metrics.json", "r", encoding="utf-8") as file:
        results = [json.load(file)]

    memory = None
    if uses_replay:
        # Fill the buffer ONCE from Stage 1 data (the only data available then).
        memory = ClassBalancedMemory(MEMORY_SIZE, seed=seed)
        memory.update(stages[0]["train"], stages[0]["new_classes"])
        print(f"Replay memory after Stage 1: {memory.summary()}")

    for stage_index in range(1, len(stages)):
        stage = stages[stage_index]
        stage_number = stage_index + 1

        old_classes = [str(c) for c in stages[stage_index - 1]["seen_classes"]]
        new_classes = [str(c) for c in stage["new_classes"]]
        seen_classes = [str(c) for c in stage["seen_classes"]]

        print("\n" + "=" * 60)
        print(f"[{name} | seed {seed}] Stage {stage_number}")
        print(f"Previously seen categories: {len(old_classes)}")
        print(f"New categories: {len(new_classes)}")
        print(f"Total categories: {len(seen_classes)}")

        stage_new_data = stage["train"].copy()  # current-stage data only
        stage_train = stage_new_data
        stage_test = stage["test"].copy()

        # Classes known before this stage: their label IDs are 0..num_old-1,
        # new classes get IDs >= num_old (used by ACE to tell them apart).
        num_old = len(label_to_id)

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
            # Replay ONLY from the persistent buffer, never from the dataset.
            memory_df = memory.as_dataframe()
            replay_count = len(memory_df)
            stage_train = pd.concat(
                [stage_new_data, memory_df], ignore_index=True)
            balance_alpha = alpha
            print(f"Replay examples: {replay_count} ({memory.summary()})")

        print(f"Training examples: {len(stage_train)}")
        print(f"Evaluation examples: {len(stage_test)}")

        train_model(
            model,
            make_loader(
                stage_train, label_to_id, tokenizer, shuffle=True,
                balance_alpha=balance_alpha),
            teacher=teacher,
            num_old=num_old,
            use_ace=uses_ace,
            lambda_kd=lambda_kd,
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

        output_dir = CHECKPOINT_ROOT / tag / f"stage{stage_number}_{name}"
        save_checkpoint(model, tokenizer, label_to_id, metrics, output_dir,
                        memory=memory, save_model=SAVE_STAGE_CHECKPOINTS)
        results.append(metrics)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_joint(spec, stages, tokenizer, seed):
    """Upper bound: for each stage, train from scratch on ALL data seen so far.

    This is not continual learning (it keeps every old example), so it shows
    the best accuracy this model and training recipe can reach on the same
    classes. The gap between it and a replay strategy is the room left to
    improve.
    """
    set_seed(seed)
    print(f"\nUsing device: {DEVICE}")
    print(f"Experiment: {spec['name']} (joint upper bound, seed={seed})")

    results = []
    for stage_index in range(len(stages)):
        stage_number = stage_index + 1
        stage = stages[stage_index]

        seen_classes = [str(c) for c in stage["seen_classes"]]
        new_classes = [str(c) for c in stage["new_classes"]]
        old_classes = (
            [str(c) for c in stages[stage_index - 1]["seen_classes"]]
            if stage_index > 0 else []
        )

        # Each stage's "train" holds only that stage's new classes.
        train_df = pd.concat(
            [stages[i]["train"] for i in range(stage_index + 1)],
            ignore_index=True,
        )
        if set(train_df["label"].astype(str)) != set(seen_classes):
            raise ValueError(
                f"Joint training data at Stage {stage_number} does not match "
                "the seen classes; check what stage['train'] contains.")

        label_to_id = {category: index
                       for index, category in enumerate(seen_classes)}

        print("\n" + "=" * 60)
        print(f"[joint | seed {seed}] Stage {stage_number}")
        print(f"Categories: {len(seen_classes)} | "
              f"Training examples: {len(train_df)} | "
              f"Evaluation examples: {len(stage['test'])}")

        model = AdaptiveDeskClassifier(num_labels=len(label_to_id))
        model.to(DEVICE)
        train_model(model, make_loader(
            train_df, label_to_id, tokenizer, shuffle=True))

        metrics = evaluate_model(
            model, stage["test"].copy(), label_to_id, tokenizer,
            old_classes, new_classes
        )
        metrics.update({
            "stage": stage_number,
            "strategy": "joint",
            "training_examples": int(len(train_df)),
            "replay_examples": 0,
        })
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        results.append(metrics)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def run_experiment(spec, stages, tokenizer, seed, mode):
    """Run one experiment for one seed and save its per-stage results."""
    name = spec["name"]
    tag = f"{mode}_seed{seed}"
    results_path = RESULTS_ROOT / tag / f"results_{name}.csv"

    if RESUME and results_path.exists():
        print(f"\nSkipping {name} (seed {seed}): found {results_path}")
        return pd.read_csv(results_path)

    if spec["strategy"] == "joint":
        results = run_joint(spec, stages, tokenizer, seed)
    else:
        results = run_incremental(spec, stages, tokenizer, seed, tag)

    frame = pd.DataFrame(results).sort_values("stage")
    frame["experiment"] = name
    frame["strategy"] = spec["strategy"]
    frame["seed"] = seed
    frame["eval_mode"] = mode

    results_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    return frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run continual-learning experiments over several seeds.")
    parser.add_argument(
        "--mode", choices=["val", "test"], default=EVAL_MODE,
        help="val: choose hyperparameters on a validation split. "
             "test: final report on the real test set.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=SEEDS,
        help="seeds to run (default: %(default)s)")
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="run only the experiments with these names")
    return parser.parse_args()


def main():
    args = parse_args()
    names = [spec["name"] for spec in EXPERIMENTS]
    if len(set(names)) != len(names):
        raise ValueError("Experiment names must be unique.")

    experiments = EXPERIMENTS
    if args.only is not None:
        unknown = set(args.only) - set(names)
        if unknown:
            raise ValueError(f"Unknown experiment names: {sorted(unknown)}")
        experiments = [s for s in EXPERIMENTS if s["name"] in args.only]

    if args.mode == "test":
        print("TEST mode: run this only for the final report, with the "
              "hyperparameters you chose in val mode.")

    train_df, test_df = load_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    frames = []
    for seed in args.seeds:
        print("\n" + "#" * 60)
        print(f"Seed {seed} ({args.mode} mode)")

        stages = build_stages(train_df, test_df, seed, args.mode)
        ensure_stage_one(stages, tokenizer, seed, f"{args.mode}_seed{seed}")

        for spec in experiments:
            frames.append(
                run_experiment(spec, stages, tokenizer, seed, args.mode))

    # ---- summaries across seeds ------------------------------------------
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(RESULTS_ROOT / f"all_results_{args.mode}.csv", index=False)

    per_run = per_run_metrics(combined)
    per_run.to_csv(RESULTS_ROOT / f"per_run_{args.mode}.csv", index=False)

    summary = summarize_runs(per_run)
    summary.to_csv(RESULTS_ROOT / f"summary_{args.mode}.csv", index=False)

    print("\n" + "=" * 60)
    print(f"Summary over seeds {args.seeds} ({args.mode} mode), in %")
    print("mean acc / old / new = averages over stages 2 onwards")
    print(format_summary(summary))

    if REFERENCE_EXPERIMENT in set(per_run["experiment"]):
        diffs = paired_differences(per_run, REFERENCE_EXPERIMENT)
        diffs.to_csv(
            RESULTS_ROOT / f"paired_diffs_{args.mode}.csv", index=False)
        shown = diffs[diffs["metric"].isin(
            ["mean_accuracy", "mean_old_accuracy", "mean_new_accuracy"])]
        print(f"\nPaired differences vs '{REFERENCE_EXPERIMENT}' "
              "(positive = better; wins = seeds where it was better)")
        print(shown.to_string(index=False))
    else:
        print(f"\n(No paired differences: '{REFERENCE_EXPERIMENT}' "
              "was not run.)")


if __name__ == "__main__":
    main()
