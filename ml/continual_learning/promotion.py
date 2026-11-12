"""Promotion gate: decide whether a new model version is safe to serve.

Pure Python (no torch), so the same rule can be reused by the API and shown in
the UI. The gate only reports; promoting stays a deliberate step in the
ModelRegistry.
"""

from dataclasses import dataclass


@dataclass
class GateSettings:
    # Allowed fall in accuracy on EXISTING categories compared with the base
    # model, measured on the same held-out tickets (0.03 = 3 points).
    max_old_accuracy_drop: float = 0.03
    # The new categories must actually have been learned.
    min_new_accuracy: float = 0.7
    # Without enough held-out evidence the gate fails instead of guessing.
    min_old_eval_examples: int = 20
    min_new_eval_examples: int = 20


def weak_categories(results, settings):
    """New categories that cannot be admitted yet, with the reason.

    `results` is a list of {"category", "n_eval", "accuracy"}.
    """
    weak = {}
    for row in results:
        if row["n_eval"] < settings.min_new_eval_examples:
            weak[row["category"]] = (
                f"only {row['n_eval']} held-out tickets; at least "
                f"{settings.min_new_eval_examples} are needed to judge it")
        elif row["accuracy"] < settings.min_new_accuracy:
            weak[row["category"]] = (
                f"accuracy {row['accuracy']:.0%} on {row['n_eval']} held-out "
                f"tickets; at least {settings.min_new_accuracy:.0%} is needed")
    return weak


def evaluate_gate(base_old_accuracy, new_old_accuracy, n_old_eval,
                  new_category_results, settings):
    """Return {"passed": bool, "checks": [...]} for a candidate version."""
    checks = []

    # 1. Existing categories must not get worse.
    if (base_old_accuracy is None or new_old_accuracy is None
            or n_old_eval < settings.min_old_eval_examples):
        checks.append({
            "name": "old_categories_not_worse",
            "passed": False,
            "value": None,
            "limit": settings.max_old_accuracy_drop,
            "detail": (
                f"Only {n_old_eval} held-out examples of existing categories; "
                f"need at least {settings.min_old_eval_examples} to judge."),
        })
    else:
        drop = base_old_accuracy - new_old_accuracy
        checks.append({
            "name": "old_categories_not_worse",
            "passed": bool(drop <= settings.max_old_accuracy_drop),
            "value": float(drop),
            "limit": settings.max_old_accuracy_drop,
            "detail": (
                f"Accuracy on existing categories went from "
                f"{base_old_accuracy:.3f} to {new_old_accuracy:.3f} "
                f"on {n_old_eval} held-out tickets."),
        })

    # 2. Every new category must be learned (skipped when there are none).
    if new_category_results:
        weak = weak_categories(new_category_results, settings)
        if weak:
            detail = "; ".join(f"{c}: {r}" for c, r in weak.items())
        else:
            worst = min(row["accuracy"] for row in new_category_results)
            detail = (f"All {len(new_category_results)} new categories reach "
                      f"{settings.min_new_accuracy:.0%} (lowest {worst:.0%}).")
        checks.append({
            "name": "new_categories_learned",
            "passed": not weak,
            "value": None,
            "limit": settings.min_new_accuracy,
            "detail": detail,
        })

    return {"passed": all(check["passed"] for check in checks),
            "checks": checks}
