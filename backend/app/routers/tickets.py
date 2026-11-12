"""Tickets: classify new ones, list the queue, confirm labels."""

import random
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.deps import get_classifier, get_db
from backend.app.models import Category, Ticket, utcnow
from backend.app.schemas import (
    LabelRequest,
    PredictOut,
    PredictionItem,
    TicketCreate,
    TicketList,
    TicketOut,
)
from backend.app.serializers import ticket_out

router = APIRouter(tags=["tickets"])

TICKET_STATUSES = {"needs_review", "auto_labeled", "confirmed"}


@router.post("/tickets", response_model=TicketOut, status_code=201)
def create_ticket(body: TicketCreate, db: Session = Depends(get_db),
                  classifier=Depends(get_classifier)):
    """Classify an incoming ticket and route it.

    Confident predictions are labelled automatically; the rest (and a small
    random spot-check share of the confident ones) go to the review queue.
    """
    prediction, version = classifier.predict(body.text, settings.top_k)
    ticket = Ticket(text=body.text, status="needs_review")

    if prediction is not None:
        ticket.model_version = version
        ticket.predicted_label = prediction.label
        ticket.confidence = prediction.confidence
        ticket.top_predictions = [
            {"label": label, "confidence": confidence}
            for label, confidence in prediction.top]

        if not prediction.needs_review:
            category = db.scalar(
                select(Category).where(Category.name == prediction.label))

            if category is not None:
                if random.random() < settings.spot_check_rate:
                    ticket.is_spot_check = True   # stays in the review queue
                else:
                    ticket.status = "auto_labeled"
                    ticket.category = category

    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket_out(ticket)


@router.post("/predict", response_model=PredictOut)
def predict(body: TicketCreate, classifier=Depends(get_classifier)):
    """Playground: classify text without saving a ticket."""
    prediction, version = classifier.predict(body.text, settings.top_k)
    if prediction is None:
        return PredictOut(model_version=None, label=None, confidence=None,
                          top=[], needs_review=True)
    return PredictOut(
        model_version=version,
        label=prediction.label,
        confidence=prediction.confidence,
        top=[PredictionItem(label=label, confidence=confidence)
             for label, confidence in prediction.top],
        needs_review=prediction.needs_review,
    )


@router.get("/tickets", response_model=TicketList)
def list_tickets(
    status: Optional[str] = None,
    sort: str = Query("newest", pattern="^(newest|lowest_confidence)$"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    filters = []
    if status is not None:
        if status not in TICKET_STATUSES:
            raise HTTPException(
                422, f"status must be one of {sorted(TICKET_STATUSES)}")
        filters.append(Ticket.status == status)

    query = select(Ticket).where(*filters)
    if sort == "lowest_confidence":
        # Tickets without a prediction first, then the least confident.
        query = query.order_by(Ticket.confidence.is_(None).desc(),
                               Ticket.confidence.asc(), Ticket.id.asc())
    else:
        query = query.order_by(Ticket.id.desc())

    total = db.scalar(select(func.count()).select_from(Ticket).where(*filters))
    tickets = db.scalars(query.limit(limit).offset(offset)).all()
    return TicketList(items=[ticket_out(t) for t in tickets], total=total)


@router.get("/tickets/{ticket_id}", response_model=TicketOut)
def get_ticket(ticket_id: int, db: Session = Depends(get_db)):
    ticket = db.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(404, "Ticket not found.")
    return ticket_out(ticket)


@router.post("/tickets/{ticket_id}/label", response_model=TicketOut)
def label_ticket(ticket_id: int, body: LabelRequest,
                 db: Session = Depends(get_db)):
    """A human confirms a category, or proposes a new one.

    A proposed category is only a label until an admin approves it; its tickets
    are not used for training before that.
    """
    ticket = db.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(404, "Ticket not found.")
    if ticket.consumed_by_version is not None:
        raise HTTPException(
            409, f"This ticket was already used by model "
            f"{ticket.consumed_by_version} and can no longer be relabelled.")

    if body.category_id is not None:
        category = db.get(Category, body.category_id)
        if category is None:
            raise HTTPException(404, "Category not found.")
        if category.status not in ("approved", "proposed"):
            raise HTTPException(
                409, f"Category '{category.name}' is {category.status}.")
    else:
        name = body.new_category.name.strip()
        if not name:
            raise HTTPException(422, "Category name cannot be empty.")

        category = db.scalar(select(Category).where(
            func.lower(Category.name) == name.lower()))
        if category is None:
            category = Category(
                name=name, description=body.new_category.description,
                status="proposed")
            db.add(category)
            db.flush()
        elif category.status not in ("approved", "proposed"):
            raise HTTPException(
                409, f"The name '{name}' belongs to a {category.status} "
                "category; choose another name.")

    ticket.category_id = category.id
    ticket.status = "confirmed"
    ticket.reviewed_at = utcnow()
    db.commit()
    db.refresh(ticket)
    return ticket_out(ticket)
