"""Training: what a run would use, start a run, follow its progress."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.deps import get_classifier, get_db, get_jobs
from backend.app.models import TrainingJob
from backend.app.schemas import JobOut, TrainingPlan
from backend.app.serializers import job_out
from backend.app.services import JobConflict, NothingToTrain, training_plan

router = APIRouter(prefix="/training", tags=["training"])


@router.get("/plan", response_model=TrainingPlan)
def get_plan(db: Session = Depends(get_db),
             classifier=Depends(get_classifier)):
    """What the next run would use, and which categories are still collecting."""
    plan = training_plan(db, classifier)
    return TrainingPlan(
        can_start=plan["can_start"],
        reason=plan["reason"],
        ready_categories=plan["ready"],
        collecting=plan["collecting"],
        min_tickets_new_category=settings.min_tickets_new_category,
        n_tickets=len(plan["ticket_ids"]),
    )


@router.post("/jobs", response_model=JobOut, status_code=202)
def start_job(db: Session = Depends(get_db), jobs=Depends(get_jobs)):
    """Start one incremental training run in the background."""
    try:
        job = jobs.start(db)
    except JobConflict as error:
        raise HTTPException(409, str(error))
    except NothingToTrain as error:
        raise HTTPException(409, str(error))
    return job_out(job)


@router.get("/jobs", response_model=list[JobOut])
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.scalars(
        select(TrainingJob).order_by(TrainingJob.id.desc()).limit(20)).all()
    return [job_out(job) for job in jobs]


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(TrainingJob, job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return job_out(job, include_report=True)
