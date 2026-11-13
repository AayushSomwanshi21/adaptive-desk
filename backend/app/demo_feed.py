"""Send demo tickets of a held-back category through the running API.

Acts as the support agent: every ticket is classified by the API, then labelled
with the new category (which creates it as "proposed" on the first ticket).

    python -m backend.app.demo_feed --list
    python -m backend.app.demo_feed --category "<name>" --count 45
    python -m backend.app.demo_feed --category "<name>" --count 45 --approve

A new category needs about 40 confirmed tickets before it can be trained, and
an admin must approve it (use --approve or POST /categories/{id}/approve).
"""

import argparse
from pathlib import Path

import httpx
import pandas as pd

from backend.app.config import ROOT

DEFAULT_DEMO_PATH = ROOT / "backend/data/demo_tickets.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--demo-file", type=Path, default=DEFAULT_DEMO_PATH)
    parser.add_argument("--category", help="category to feed")
    parser.add_argument("--count", type=int, default=45)
    parser.add_argument("--skip", type=int, default=0,
                        help="skip this many tickets of the category first")
    parser.add_argument("--list", action="store_true",
                        help="list the categories available in the demo file")
    parser.add_argument("--approve", action="store_true",
                        help="approve the category after feeding it")
    args = parser.parse_args()

    if not args.demo_file.exists():
        raise SystemExit(
            f"{args.demo_file} not found; run backend.app.bootstrap with "
            "--classes N to hold some categories back.")

    demo = pd.read_csv(args.demo_file, dtype=str).fillna("")
    counts = demo["label"].value_counts()

    if args.list or not args.category:
        print("Categories available in the demo file (tickets):")
        for name, count in counts.items():
            print(f"  {name}  ({count})")
        return

    rows = demo[demo["label"] == args.category].iloc[
        args.skip:args.skip + args.count]
    if rows.empty:
        raise SystemExit(f"No demo tickets for '{args.category}'. Use --list.")

    category_id = None
    category_status = None
    with httpx.Client(base_url=args.api, timeout=60) as client:
        for number, text in enumerate(rows["text"], start=1):
            ticket = client.post("/tickets", json={"text": text})
            ticket.raise_for_status()
            ticket_id = ticket.json()["id"]

            labelled = client.post(f"/tickets/{ticket_id}/label", json={
                "new_category": {
                    "name": args.category,
                    "description": "Demo category fed by demo_feed."},
            })
            labelled.raise_for_status()
            category_id = labelled.json()["category"]["id"]
            category_status = labelled.json()["category"]["status"]

            if number % 10 == 0:
                print(f"  sent {number}/{len(rows)}")

        print(f"Sent {len(rows)} tickets to category '{args.category}' "
              f"(id {category_id}).")

        if category_status == "proposed" and args.approve:
            client.post(
                f"/categories/{category_id}/approve").raise_for_status()
            print("Category approved.")
        elif category_status == "proposed":
            print(f"Approve it with: POST /categories/{category_id}/approve")
        else:
            print(
                f"Category is already {category_status}; nothing to approve.")

        plan = client.get("/training/plan").json()
        print(f"Training plan: can_start={plan['can_start']}, "
              f"reason={plan['reason']}")


if __name__ == "__main__":
    main()
