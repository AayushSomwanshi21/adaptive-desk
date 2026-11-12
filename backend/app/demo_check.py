"""Check the active model on demo tickets of a category that were NOT fed to it.

demo_feed sends the first N tickets of a category (and the stage update trains
on or holds out those); the rest of that category in the demo file are fresh
tickets, which give a better accuracy estimate than the handful of held-out
ones.

    python -m backend.app.demo_check --category "<name>" --skip 45
"""

import argparse
from collections import Counter
from pathlib import Path

import httpx
import pandas as pd

from backend.app.config import ROOT

DEFAULT_DEMO_PATH = ROOT / "backend/data/demo_tickets.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--demo-file", type=Path, default=DEFAULT_DEMO_PATH)
    parser.add_argument("--category", required=True)
    parser.add_argument("--skip", type=int, default=45,
                        help="how many tickets demo_feed already sent")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    demo = pd.read_csv(args.demo_file, dtype=str).fillna("")
    rows = demo[demo["label"] == args.category].iloc[
        args.skip:args.skip + args.limit]
    if rows.empty:
        raise SystemExit("No unseen tickets left for that category.")

    correct = 0
    auto_correct = auto_wrong = 0
    confidences = []
    mistakes = Counter()

    with httpx.Client(base_url=args.api, timeout=60) as client:
        for text in rows["text"]:
            response = client.post("/predict", json={"text": text})
            response.raise_for_status()
            result = response.json()

            is_correct = result["label"] == args.category
            correct += is_correct
            confidences.append(result["confidence"] or 0.0)
            if not is_correct:
                mistakes[result["label"]] += 1

            if not result["needs_review"]:   # would be auto-labelled
                if is_correct:
                    auto_correct += 1
                else:
                    auto_wrong += 1

    total = len(rows)
    auto = auto_correct + auto_wrong
    print(f"Model {result['model_version']} on {total} unseen "
          f"'{args.category}' tickets")
    print(f"  correct: {correct}/{total} ({correct / total:.0%})")
    print(f"  average confidence: {sum(confidences) / total:.2f}")
    print(f"  would be auto-labelled: {auto}/{total} "
          f"({auto_wrong} of those wrongly)")
    print(f"  sent to review: {total - auto}/{total}")
    if mistakes:
        print("  most common wrong predictions:")
        for label, count in mistakes.most_common(3):
            print(f"    {label}: {count}")


if __name__ == "__main__":
    main()
