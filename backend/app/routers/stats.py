"""Gives snapshot of the ticket pipeline for the dashboard."""

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.deps import get_classifier, get_db
from backend.app.models import Category, Ticket

router = APIRouter(tags=["stats"])


@router.get("/stats")
def get_stats(db: Session = Depends(get_db),
              classifier=Depends(get_classifier)):
    by_status = dict(db.execute(
        select(Ticket.status, func.count()).group_by(Ticket.status)).all())
    categories = dict(db.execute(
        select(Category.status, func.count()).group_by(Category.status)).all())

    classified = db.scalar(select(func.count()).select_from(Ticket).where(
        Ticket.model_version.is_not(None))) or 0
    auto_labeled = by_status.get("auto_labeled", 0)
    average_confidence = db.scalar(select(func.avg(Ticket.confidence)).where(
        Ticket.confidence.is_not(None)))

    # Of the spot-checked tickets a human has reviewed, how often did the
    # human agree with the model? This estimates how trustworthy the
    # auto-labelled tickets are.
    spot_checks = db.execute(
        select(Ticket.predicted_label, Category.name)
        .join(Category, Ticket.category_id == Category.id)
        .where(Ticket.is_spot_check.is_(True), Ticket.status == "confirmed")
    ).all()
    agreement = (
        sum(1 for predicted, chosen in spot_checks if predicted == chosen)
        / len(spot_checks) if spot_checks else None)

    return {
        "model": classifier.info(),
        "tickets": {
            "total": sum(by_status.values()),
            "by_status": by_status,
            "review_queue": by_status.get("needs_review", 0),
            "auto_label_rate": (auto_labeled / classified
                                if classified else None),
            "average_confidence": average_confidence,
        },
        "spot_checks": {"reviewed": len(spot_checks),
                        "agreement": agreement},
        "categories": categories,
    }
