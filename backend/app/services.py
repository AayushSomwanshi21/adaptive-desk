"""Services that sit between the API, the database and the ML code."""

import json
import threading
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd
from sqlalchemy import select, update

from backend.app.config import settings
from backend.app.models import Category, Ticket, TrainingJob, utcnow


class JobConflict(Exception):
    """A training job is already queued or running."""


class NothingToTrain(Exception):
    """There is no category ready to be trained."""


# categories
def sync_categories(db, labels):
    """Make sure every category the model knows has a database row."""
    existing = set(db.scalars(select(Category.name)))
    for label in sorted(labels):
        if label not in existing:
            db.add(Category(name=label, status="approved"))
    db.commit()


# ---- what would a training run use? ----------------------------------------
def training_plan(db, classifier):
    """Which confirmed tickets would go into the next training run.

    Rules:
      * only tickets a human confirmed, in APPROVED categories, that no earlier
        run has used yet;
      * tickets of categories the model already knows are always included;
      * a NEW category is included only once it has enough tickets to be both
        trained on and judged (settings.min_tickets_new_category); until then
        it is "collecting examples";
      * a run only starts when at least one new category is ready.
    """
    minimum = settings.min_tickets_new_category
    plan = {"can_start": False, "reason": None, "ready": [],
            "collecting": [], "refresh": [], "ticket_ids": [], "rows": []}

    if classifier.version is None:
        plan["reason"] = ("No model is active yet. Run "
                          "`python -m backend.app.bootstrap` first.")
        return plan

    found = db.execute(
        select(Ticket.id, Ticket.text, Category.name)
        .join(Category, Ticket.category_id == Category.id)
        .where(Ticket.status == "confirmed",
               Ticket.consumed_by_version.is_(None),
               Category.status == "approved")
    ).all()

    by_category = defaultdict(list)
    for ticket_id, text, name in found:
        by_category[name].append((ticket_id, text))

    model_labels = classifier.labels
    for name, items in sorted(by_category.items()):
        if name in model_labels:
            if len(items) >= settings.min_tickets_existing_category:
                plan["refresh"].append({
                    "name": name, "tickets": len(items),
                    "needed": settings.min_tickets_existing_category})
            chosen = items
        elif len(items) >= minimum:
            plan["ready"].append(
                {"name": name, "tickets": len(items), "needed": minimum})
            chosen = items
        else:
            plan["collecting"].append(
                {"name": name, "tickets": len(items), "needed": minimum})
            continue

        for ticket_id, text in chosen:
            plan["ticket_ids"].append(ticket_id)
            plan["rows"].append((text, name))

    if plan["ready"] or plan['refresh']:
        plan["can_start"] = True
    elif plan["collecting"]:
        plan["reason"] = (
            f"No new category has {minimum} confirmed tickets yet.")
    else:
        plan["reason"] = (
            "There are no approved new categories with confirmed tickets.")
    return plan


def mark_consumed(db, version):
    """After `version` is promoted, mark the tickets its training run used.

    Those tickets were either trained on or held out for evaluation, so they
    must not be sent to training again.
    """
    job = db.scalar(select(TrainingJob).where(
        TrainingJob.version == version, TrainingJob.status == "succeeded"))
    if job is None:
        return 0

    ids = job.ticket_ids or []
    for start in range(0, len(ids), 500):
        db.execute(
            update(Ticket)
            .where(Ticket.id.in_(ids[start:start + 500]))
            .values(consumed_by_version=version))
    db.commit()
    return len(ids)


# ---- training jobs ---------------------------------------------------------
def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


class JobRunner:
    """Runs one training job at a time in a background thread.

    It runs inside the web process, so the model keeps answering requests (more
    slowly) while it trains.
    """

    def __init__(self, session_factory, registry, classifier):
        self._session_factory = session_factory
        self.registry = registry
        self.classifier = classifier
        self._start_lock = threading.Lock()

    def mark_interrupted(self):
        """Jobs left running by a crash or restart can never finish."""
        with self._session_factory() as db:
            for job in db.scalars(select(TrainingJob).where(
                    TrainingJob.status.in_(["queued", "running"]))):
                job.status = "failed"
                job.error = "Interrupted by a server restart."
                job.finished_at = utcnow()
            db.commit()

    def start(self, db):
        with self._start_lock:
            busy = db.scalar(select(TrainingJob).where(
                TrainingJob.status.in_(["queued", "running"])))
            if busy is not None:
                raise JobConflict(
                    f"Training job {busy.id} is still {busy.status}.")

            plan = training_plan(db, self.classifier)
            if not plan["can_start"]:
                raise NothingToTrain(plan["reason"])

            job = TrainingJob(
                status="queued",
                base_version=self.registry.active(),
                ticket_ids=plan["ticket_ids"],
            )
            db.add(job)
            db.commit()
            db.refresh(job)

            threading.Thread(
                target=self._run, args=(job.id, plan["rows"]), daemon=True
            ).start()
        return job

    def _run(self, job_id, rows):
        # Imported here so the web app can start (and show a clear error)
        # even if a heavy ML dependency fails to import.
        from ml.continual_learning.stage_update import (
            UpdateSettings,
            run_stage_update,
        )

        with self._session_factory() as db:
            job = db.get(TrainingJob, job_id)
            try:
                job.status = "running"
                job.started_at = utcnow()
                db.commit()

                tickets = pd.DataFrame(rows, columns=["text", "label"])
                update_settings = UpdateSettings(
                    holdout_fraction=settings.holdout_fraction,
                    model_name=settings.model_name)

                report = run_stage_update(
                    self.registry, tickets, update_settings,
                    base_version=job.base_version)

                job.version = report["version"]
                job.report = json.loads(
                    json.dumps(report, default=_json_default))
                job.status = "succeeded"
            except Exception as error:  # the job must always end up finished
                traceback.print_exc()
                db.rollback()
                job = db.get(TrainingJob, job_id)
                job.status = "failed"
                job.error = f"{type(error).__name__}: {error}"
            finally:
                job.finished_at = utcnow()
                db.commit()
