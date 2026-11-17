"""Plot continual-learning experiment results and saved model-version metrics.

Run from the repository root:

    python -m ml.evaluation.plot_results

The script writes figures to ``ml/experiments/results/plots/final_results``
by default.
Seed comparisons use ``all_results_test.csv`` when available (or a ``seed``
column in the per-strategy CSVs). Category-wise forgetting requires
``category_accuracy_all_stages.csv`` with per-category results for each
strategy; aggregate accuracy CSVs alone are not sufficient to calculate it.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "ml" / "experiments" / "results"
DEFAULT_REGISTRY_DIR = ROOT / "ml" / "models" / "registry"

RESULT_FILES = {
    "baseline": "results_baseline_all_stages.csv",
    "replay_kd": "results_replay_kd_a0.5_l0.25.csv",
    "replay": "results_replay_all_stages.csv",
}

SEED_RESULT_FILES = {
    "baseline": "results_baseline.csv",
    "replay_kd": "results_replay_kd_a0.5_l0.25.csv",
    "replay": "results_replay.csv",
}

FALLBACK_RESULT_FILES = {
    "replay_kd": ("results_replay_kd_all_stages_500.csv",),
}

METRICS = (
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("old_category_accuracy", "Old-category accuracy"),
    ("new_category_accuracy", "New-category accuracy"),
)

SEED_VALUES = (42, 43, 44, 45, 46)
VERSION_NAMES = ("v0001", "v0002", "v0003", "v0004")


def _read_csv(path):
    frame = pd.read_csv(path)
    if "stage" not in frame.columns:
        raise ValueError(f"{path} must contain a 'stage' column.")

    frame["stage"] = pd.to_numeric(frame["stage"], errors="raise")
    for metric, _ in METRICS:
        if metric in frame.columns:
            frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    return frame


def _read_stage_results(results_dir, strategy):
    """Read an aggregate CSV or combine the runner's per-seed CSVs."""
    aggregate_path = results_dir / RESULT_FILES[strategy]
    if aggregate_path.is_file():
        return _read_csv(aggregate_path).sort_values("stage")

    seed_paths = sorted(results_dir.glob(
        f"*/{SEED_RESULT_FILES[strategy]}"))
    seed_frames = []
    for path in seed_paths:
        match = re.search(r"(?:val|test)_seed(\d+)$", path.parent.name)
        if not match:
            continue
        frame = _read_csv(path)
        frame["seed"] = int(match.group(1))
        frame["eval_mode"] = (
            "val" if path.parent.name.startswith("val_") else "test")
        seed_frames.append(frame)

    if seed_frames:
        print(
            f"Aggregate file {aggregate_path.name} not found; combining "
            f"{len(seed_frames)} per-seed result files."
        )
        return pd.concat(seed_frames, ignore_index=True).sort_values(
            ["seed", "stage"])

    for fallback_name in FALLBACK_RESULT_FILES.get(strategy, ()):
        fallback_path = results_dir / fallback_name
        if fallback_path.is_file():
            print(
                f"Using {fallback_name} as the available {strategy} "
                "aggregate result."
            )
            return _read_csv(fallback_path).sort_values("stage")

    raise FileNotFoundError(
        f"Could not find results for '{strategy}'. Looked for "
        f"{aggregate_path}, per-seed files named "
        f"{SEED_RESULT_FILES[strategy]} under {results_dir}, and configured "
        "aggregate fallbacks."
    )


def _mean_and_std(frame, metric):
    values = frame[["stage", metric]].dropna()
    if values.empty:
        return None

    grouped = values.groupby("stage")[metric]
    means = grouped.mean().sort_index()
    stds = grouped.std().reindex(means.index).fillna(0.0)
    return means.index.to_numpy(), means.to_numpy(), stds.to_numpy()


def _plot_metric(ax, frame, metric, label):
    if metric not in frame.columns:
        return False

    series = _mean_and_std(frame, metric)
    if series is None:
        return False

    stages, means, stds = series
    ax.plot(stages, means * 100, marker="o", label=label)
    has_multiple_seeds = (
        "seed" in frame.columns and frame["seed"].nunique() > 1
    )
    if has_multiple_seeds:
        ax.fill_between(
            stages,
            (means - stds) * 100,
            (means + stds) * 100,
            alpha=0.15,
        )
    return True


def _format_metric_axis(ax, title, ylabel):
    ax.set_title(title)
    ax.set_xlabel("Stage")
    ax.set_ylabel(f"{ylabel} (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_strategy_results(results, output_dir):
    """Make individual per-strategy plots and three-strategy comparisons."""
    for strategy, frame in results.items():
        for metric, title in METRICS:
            if metric not in frame.columns or frame[metric].isna().all():
                print(f"Skipping {strategy} {metric}: column/data unavailable.")
                continue

            fig, ax = plt.subplots(figsize=(8, 5))
            _plot_metric(ax, frame, metric, strategy.replace("_", " "))
            _format_metric_axis(ax, f"{title} - {strategy}", title)
            fig.tight_layout()
            fig.savefig(output_dir / f"{strategy}_{metric}.png", dpi=160)
            plt.close(fig)

    for metric, title in METRICS:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        plotted = False
        for strategy, frame in results.items():
            plotted |= _plot_metric(
                ax, frame, metric, strategy.replace("_", " "))
        if not plotted:
            plt.close(fig)
            print(f"Skipping comparison for {metric}: no data available.")
            continue

        _format_metric_axis(ax, f"{title} Across Strategies", title)
        fig.tight_layout()
        fig.savefig(output_dir / f"strategy_comparison_{metric}.png", dpi=160)
        plt.close(fig)


def _load_seed_results(results_dir, strategy_results):
    combined_path = results_dir / "all_results_test.csv"
    if combined_path.is_file():
        frame = pd.read_csv(combined_path)
        if not {"stage", "seed"}.issubset(frame.columns):
            raise ValueError(
                f"{combined_path} must contain 'stage' and 'seed' columns."
            )
        if "experiment" not in frame.columns:
            raise ValueError(
                f"{combined_path} must contain an 'experiment' column."
            )
        return frame

    seeded = []
    for strategy, filename in SEED_RESULT_FILES.items():
        for path in sorted(results_dir.glob(f"*/{filename}")):
            match = re.search(r"(?:val|test)_seed(\d+)$", path.parent.name)
            if not match:
                continue
            part = _read_csv(path)
            part["seed"] = int(match.group(1))
            part["experiment"] = strategy
            part["eval_mode"] = (
                "val" if path.parent.name.startswith("val_") else "test")
            seeded.append(part)
    if seeded:
        return pd.concat(seeded, ignore_index=True)

    combined_val_path = results_dir / "all_results_val.csv"
    if combined_val_path.is_file():
        frame = pd.read_csv(combined_val_path)
        if not {"stage", "seed", "experiment"}.issubset(frame.columns):
            raise ValueError(
                f"{combined_val_path} must contain 'stage', 'seed', and "
                "'experiment' columns."
            )
        return frame

    seeded = []
    for strategy, frame in strategy_results.items():
        if "seed" not in frame.columns:
            continue
        part = frame.copy()
        part["experiment"] = strategy
        seeded.append(part)

    return pd.concat(seeded, ignore_index=True) if seeded else pd.DataFrame()


def plot_seed_comparisons(seed_results, output_dir):
    if seed_results.empty:
        print(
            "Skipping seed plots: no seed data found. Provide "
            "all_results_test.csv or add a 'seed' column to the result CSVs."
        )
        return

    seed_results = seed_results.copy()
    seed_results["stage"] = pd.to_numeric(seed_results["stage"], errors="raise")
    seed_results["seed"] = pd.to_numeric(seed_results["seed"], errors="raise")
    for metric, _ in METRICS:
        if metric in seed_results.columns:
            seed_results[metric] = pd.to_numeric(
                seed_results[metric], errors="coerce")

    for metric, title in METRICS:
        if metric not in seed_results.columns:
            continue

        for strategy in RESULT_FILES:
            # The aggregated experiment runner records descriptive experiment
            # names; map those to the same strategy names used by this script.
            experiment_names = seed_results["experiment"].astype(str)
            if strategy == "replay_kd":
                mask = experiment_names.str.startswith("replay_kd")
            else:
                mask = experiment_names == strategy
            subset = seed_results[mask]
            if subset.empty:
                continue

            fig, ax = plt.subplots(figsize=(9, 5.5))
            plotted = False
            for seed in SEED_VALUES:
                seed_frame = subset[subset["seed"] == seed]
                if seed_frame.empty:
                    continue
                plotted |= _plot_metric(
                    ax, seed_frame, metric, f"Seed {seed}")

            if not plotted:
                plt.close(fig)
                continue
            _format_metric_axis(
                ax, f"{title} by Seed - {strategy}", title)
            fig.tight_layout()
            fig.savefig(
                output_dir / f"seed_comparison_{strategy}_{metric}.png",
                dpi=160,
            )
            plt.close(fig)


def _load_version_metrics(registry_dir):
    versions = []
    for name in VERSION_NAMES:
        path = registry_dir / "versions" / name / "metrics.json"
        if not path.is_file():
            print(f"Skipping missing model-version metrics: {path}")
            continue
        with open(path, encoding="utf-8") as file:
            metrics = json.load(file)
        versions.append((name, metrics))
    return versions


def _metric_value(metrics, *keys):
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return np.nan


def plot_version_metrics(versions, output_dir):
    if not versions:
        print("Skipping version plots: no version metrics.json files found.")
        return

    names = [name for name, _ in versions]
    version_metrics = {
        "Overall accuracy": [
            _metric_value(item, "accuracy") for _, item in versions
        ],
        "Old-category accuracy": [
            _metric_value(item, "old_accuracy", "old_category_accuracy")
            for _, item in versions
        ],
        "New-category accuracy": [
            _metric_value(item, "new_category_accuracy")
            for _, item in versions
        ],
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, values in version_metrics.items():
        if not np.isnan(values).all():
            ax.plot(names, np.asarray(values) * 100, marker="o", label=label)
    ax.set_title("Model Metrics by Version")
    ax.set_xlabel("Model version")
    ax.set_ylabel("Accuracy (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "model_version_comparison.png", dpi=160)
    plt.close(fig)


def _category_rows(versions):
    rows = []
    for version, metrics in versions:
        for item in metrics.get("per_category", []):
            if "category" not in item or "accuracy" not in item:
                continue
            rows.append({
                "version": version,
                "category": str(item["category"]),
                "accuracy": float(item["accuracy"]),
                "n_eval": int(item.get("n_eval", 0)),
            })
    return pd.DataFrame(rows, columns=[
        "version", "category", "accuracy", "n_eval"])


def _category_forgetting(category_results):
    """Compute best-prior minus current accuracy for each category/version."""
    rows = []
    for category, group in category_results.groupby("category", sort=False):
        group = group.sort_values("version")
        best_previous = None
        for row in group.itertuples(index=False):
            forgetting = (
                0.0 if best_previous is None
                else best_previous - row.accuracy
            )
            rows.append({
                "version": row.version,
                "category": category,
                "accuracy": row.accuracy,
                "forgetting": forgetting,
                "has_prior_version": best_previous is not None,
            })
            best_previous = (
                row.accuracy if best_previous is None
                else max(best_previous, row.accuracy)
            )
    return pd.DataFrame(rows)


def plot_version_categories_and_forgetting(versions, output_dir):
    categories = _category_rows(versions)
    if categories.empty:
        print(
            "Skipping category and forgetting plots: version metrics have no "
            "'per_category' entries."
        )
        return

    for version, frame in categories.groupby("version", sort=True):
        frame = frame.sort_values("accuracy")
        height = max(5.0, min(22.0, 0.24 * len(frame)))
        fig, ax = plt.subplots(figsize=(12, height))
        ax.barh(frame["category"], frame["accuracy"] * 100, color="#4472C4")
        ax.set_xlim(0, 100)
        ax.set_title(f"Category Accuracy - {version}")
        ax.set_xlabel("Accuracy (%)")
        ax.set_ylabel("Category")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"category_accuracy_{version}.png", dpi=160)
        plt.close(fig)

    forgetting = _category_forgetting(categories)
    average = (
        forgetting.groupby("version", sort=True)["forgetting"]
        .mean()
        .mul(100)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(average.index, average.values, marker="o", label="Average forgetting")
    ax.set_title("Average Forgetting Across Model Versions")
    ax.set_xlabel("Model version")
    ax.set_ylabel("Best prior accuracy - current accuracy (percentage points)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "average_forgetting_versions.png", dpi=160)
    plt.close(fig)
    forgetting.to_csv(output_dir / "version_forgetting_per_category.csv",
                      index=False)


def _normalize_strategy(value):
    value = str(value).lower()
    if value.startswith("baseline"):
        return "baseline"
    if value.startswith("replay_kd"):
        return "replay_kd"
    if value.startswith("replay"):
        return "replay"
    return value


def _load_category_stage_results(results_dir, strategy_results):
    frames = []
    category_path = results_dir / "category_accuracy_all_stages.csv"
    if category_path.is_file():
        category_frame = pd.read_csv(category_path)
        if {"stage", "strategy", "category", "accuracy"}.issubset(
                category_frame.columns):
            category_frame["strategy"] = category_frame["strategy"].map(
                _normalize_strategy)
            frames.append(category_frame)
        else:
            print(
                f"Skipping {category_path}: expected stage, strategy, "
                "category, and accuracy columns."
            )

    # Also accept per-category rows embedded in the strategy CSVs.
    for strategy, frame in strategy_results.items():
        if {"category", "accuracy"}.issubset(frame.columns):
            part = frame.copy()
            part["strategy"] = strategy
            frames.append(part)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["stage"] = pd.to_numeric(combined["stage"], errors="raise")
    combined["accuracy"] = pd.to_numeric(
        combined["accuracy"], errors="coerce")
    if "seed" not in combined.columns:
        combined["seed"] = -1
    return combined.drop_duplicates(
        subset=["stage", "strategy", "category", "seed"])


def _strategy_forgetting(category_results):
    rows = []
    keys = ["strategy", "seed", "category"]
    for (strategy, seed, category), group in category_results.groupby(
            keys, sort=False, dropna=False):
        group = group.sort_values("stage")
        best_previous = None
        for row in group.itertuples(index=False):
            forgetting = (
                0.0 if best_previous is None
                else best_previous - row.accuracy
            )
            rows.append({
                "stage": row.stage,
                "strategy": strategy,
                "seed": seed,
                "category": category,
                "forgetting": forgetting,
            })
            best_previous = (
                row.accuracy if best_previous is None
                else max(best_previous, row.accuracy)
            )
    return pd.DataFrame(rows)


def plot_strategy_forgetting(category_results, output_dir):
    if category_results.empty:
        print(
            "Skipping strategy forgetting plot: provide per-category "
            "accuracy by stage in category_accuracy_all_stages.csv. "
            "Overall accuracy CSVs cannot determine average forgetting."
        )
        return

    forgetting = _strategy_forgetting(category_results)
    available = set(forgetting["strategy"])
    missing = set(RESULT_FILES) - available
    if missing:
        print(
            "Strategy forgetting is available only for strategies with "
            f"per-category results; missing: {sorted(missing)}."
        )

    average = (
        forgetting.groupby(["stage", "strategy"], as_index=False)
        ["forgetting"].mean()
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for strategy in RESULT_FILES:
        subset = average[average["strategy"] == strategy].sort_values("stage")
        if subset.empty:
            continue
        ax.plot(
            subset["stage"], subset["forgetting"] * 100,
            marker="o", label=strategy.replace("_", " "),
        )
    ax.set_title("Average Forgetting by Strategy")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Best prior accuracy - current accuracy (percentage points)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "average_forgetting_strategies.png", dpi=160)
    plt.close(fig)
    forgetting.to_csv(output_dir / "strategy_forgetting_per_category.csv",
                      index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot experiment results and saved model-version metrics.")
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="Folder containing the experiment CSV files.")
    parser.add_argument(
        "--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR,
        help="Model registry folder containing versions/v0001..v0004.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Plot output folder (default: <results-dir>/plots/final_results).")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir / "plots" / "final_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        strategy: _read_stage_results(results_dir, strategy)
        for strategy in RESULT_FILES
    }
    evaluation_modes = {}
    for strategy, frame in results.items():
        if "eval_mode" in frame.columns:
            modes = set(frame["eval_mode"].dropna().astype(str))
        else:
            modes = set()
        evaluation_modes[strategy] = modes or {"unspecified"}
    if len(set.union(*evaluation_modes.values())) > 1:
        print(
            "Warning: strategy comparison inputs do not all identify the "
            "same evaluation mode: "
            f"{evaluation_modes}. Treat cross-strategy plots as indicative, "
            "not a controlled comparison."
        )
    plot_strategy_results(results, output_dir)

    seed_results = _load_seed_results(results_dir, results)
    plot_seed_comparisons(seed_results, output_dir)

    versions = _load_version_metrics(args.registry_dir)
    plot_version_metrics(versions, output_dir)
    plot_version_categories_and_forgetting(versions, output_dir)

    category_stage_results = _load_category_stage_results(
        results_dir, results)
    plot_strategy_forgetting(category_stage_results, output_dir)

    print(f"Plots and derived forgetting tables saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
