from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

RESULTS_DIR = Path("ml/experiments/results")
BASELINE_CSV = RESULTS_DIR / "results_baseline_all_stages.csv"
REPLAY_CSV = RESULTS_DIR / "results_replay_all_stages.csv"
FORGETTING_CSV = RESULTS_DIR / "average_forgetting_all_stages.csv"

OUTPUT_DIR = RESULTS_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

baseline = pd.read_csv(BASELINE_CSV).sort_values("stage")
replay = pd.read_csv(REPLAY_CSV).sort_values("stage")

# Stage 1 is shared: replay starts from the baseline Stage 1 checkpoint.
if 1 not in replay["stage"].values and 1 in baseline["stage"].values:
    replay_stage1 = baseline[baseline["stage"] == 1].copy()
    replay_stage1["strategy"] = "replay"
    replay = pd.concat([replay_stage1, replay],
                       ignore_index=True).sort_values("stage")

plots = [
    ("accuracy", "Overall Accuracy", "overall_accuracy.png"),
    ("macro_f1", "Macro-F1 Score", "macro_f1.png"),
    ("old_category_accuracy", "Old-Category Accuracy", "old_category_accuracy.png"),
    ("new_category_accuracy", "New-Category Accuracy", "new_category_accuracy.png"),
]

for metric, title, filename in plots:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(baseline["stage"], baseline[metric] * 100, marker="o",
            label="Sequential baseline")
    ax.plot(replay["stage"], replay[metric] * 100, marker="o",
            label="Experience Replay")
    ax.set_title(f"{title} Across Stages")
    ax.set_xlabel("Stage")
    ax.set_ylabel(f"{title} (%)")
    ax.set_xticks(range(1, 9))
    ax.set_xlim(1, 8)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.show()
    plt.close(fig)

# Plot average forgetting
forgetting = pd.read_csv(FORGETTING_CSV)
fig, ax = plt.subplots(figsize=(8, 5))

for strategy, label in [
    ("baseline", "Sequential baseline"),
    ("replay", "Experience Replay"),
]:
    strategy_df = forgetting[forgetting["strategy"]
                             == strategy].sort_values("stage")
    if not strategy_df.empty:
        ax.plot(strategy_df["stage"], strategy_df["average_forgetting"] * 100,
                marker="o", label=label)

ax.set_title("Average Forgetting Across Stages")
ax.set_xlabel("Stage")
ax.set_ylabel("Average Forgetting (%)")
ax.set_xticks(range(1, 9))
ax.set_xlim(1, 8)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "average_forgetting.png", dpi=160)
plt.show()
plt.close(fig)

print(f"Plots saved to: {OUTPUT_DIR.resolve()}")
