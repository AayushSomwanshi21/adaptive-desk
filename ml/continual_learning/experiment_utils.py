"""Pandas-only helpers for experiments: validation splits and seed summaries.

Nothing here imports torch, so it is fast to import and easy to unit-test.
"""

import numpy as np
import pandas as pd


def make_validation_split(df, fraction, seed):
    """Stratified split of `df` into (train, validation) by the "label" column.

    Every class keeps at least one example on each side (when it has two or
    more), the split is reproducible for a given seed, and the two parts never
    overlap. Use the validation part to choose hyperparameters so the real test
    set is only touched for the final report.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be strictly between 0 and 1")

    df = df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    validation_rows = []

    for _, group in df.groupby("label", sort=True):
        size = len(group)
        if size < 2:
            continue  # a single example stays in the training part

        count = int(round(size * fraction))
        count = min(max(count, 1), size - 1)
        chosen = rng.choice(group.index.to_numpy(), size=count, replace=False)
        validation_rows.extend(chosen.tolist())

    is_validation = df.index.isin(validation_rows)
    train_part = df[~is_validation].reset_index(drop=True)
    validation_part = df[is_validation].reset_index(drop=True)
    return train_part, validation_part


# Columns of the per-stage results table that the summaries read.
_STAGE_COLUMNS = [
    "accuracy",
    "old_category_accuracy",
    "new_category_accuracy",
]

# Metrics reported per run (all in percent).
RUN_METRICS = [
    "final_accuracy",
    "final_old_accuracy",
    "final_new_accuracy",
    "mean_accuracy",
    "mean_old_accuracy",
    "mean_new_accuracy",
]


def per_run_metrics(results):
    """Collapse the per-stage table into one row per (experiment, seed).

    `results` needs the columns: experiment, seed, stage, accuracy,
    old_category_accuracy, new_category_accuracy.

    "final_*" is the last stage. "mean_*" averages stages 2 onwards, because
    Stage 1 is identical for every strategy and has no old classes.
    """
    results = results.copy()
    for column in _STAGE_COLUMNS:
        results[column] = pd.to_numeric(results[column], errors="coerce")

    rows = []
    for (experiment, seed), run in results.groupby(
            ["experiment", "seed"], sort=False):
        run = run.sort_values("stage")
        incremental = run[run["stage"] >= 2]
        final = run.iloc[-1]

        rows.append({
            "experiment": experiment,
            "seed": seed,
            "final_accuracy": 100 * final["accuracy"],
            "final_old_accuracy": 100 * final["old_category_accuracy"],
            "final_new_accuracy": 100 * final["new_category_accuracy"],
            "mean_accuracy": 100 * incremental["accuracy"].mean(),
            "mean_old_accuracy": 100 * incremental[
                "old_category_accuracy"].mean(),
            "mean_new_accuracy": 100 * incremental[
                "new_category_accuracy"].mean(),
        })

    return pd.DataFrame(rows)


def summarize_runs(per_run):
    """Mean and sample standard deviation across seeds for each experiment."""
    grouped = per_run.groupby("experiment", sort=False)
    summary = grouped[RUN_METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "n_seeds", grouped.size())
    return summary.reset_index()


def paired_differences(per_run, reference):
    """Per-seed differences of every experiment against `reference`.

    Experiments share seeds (same class order, same stage-1 model), so a paired
    difference removes much of the seed-to-seed variation. `wins` counts the
    seeds where the experiment beat the reference.
    """
    reference_runs = per_run[per_run["experiment"] == reference]
    if reference_runs.empty:
        raise ValueError(f"No runs found for reference '{reference}'.")
    reference_runs = reference_runs.set_index("seed")

    rows = []
    for experiment, runs in per_run.groupby("experiment", sort=False):
        if experiment == reference:
            continue

        runs = runs.set_index("seed")
        shared = runs.index.intersection(reference_runs.index)
        if len(shared) == 0:
            continue

        for metric in RUN_METRICS:
            diff = runs.loc[shared, metric] - reference_runs.loc[
                shared, metric]
            rows.append({
                "experiment": experiment,
                "reference": reference,
                "metric": metric,
                "mean_diff": diff.mean(),
                "std_diff": diff.std(ddof=1) if len(diff) > 1 else np.nan,
                "wins": int((diff > 0).sum()),
                "n_seeds": int(len(diff)),
            })

    return pd.DataFrame(rows)


def format_summary(summary):
    """Readable "mean ± std" table for the console."""

    def cell(row, metric):
        mean = row[f"{metric}_mean"]
        std = row[f"{metric}_std"]
        if pd.isna(mean):
            return "-"
        if pd.isna(std):
            return f"{mean:.1f}"
        return f"{mean:.1f} ± {std:.1f}"

    table = pd.DataFrame({
        "experiment": summary["experiment"],
        "seeds": summary["n_seeds"],
        "final acc": [cell(r, "final_accuracy") for _, r in summary.iterrows()],
        "mean acc": [cell(r, "mean_accuracy") for _, r in summary.iterrows()],
        "mean old": [cell(r, "mean_old_accuracy") for _, r in summary.iterrows()],
        "mean new": [cell(r, "mean_new_accuracy") for _, r in summary.iterrows()],
    })
    return table.to_string(index=False)
