"""Categories: list, create, propose/approve/reject, merge."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.deps import get_classifier, get_db
from backend.app.models import Category, Ticket
from backend.app.schemas import (
    CategoryCreate,
    CategoryOut,
    CategoryUpdate,
    MergeRequest,
)
from backend.app.serializers import category_out

router = APIRouter(prefix="/categories", tags=["categories"])

CATEGORY_STATUSES = {"proposed", "approved", "rejected", "merged"}


def _available_counts(db):
    """Confirmed tickets per category that no training run has used yet."""
    rows = db.execute(
        select(Ticket.category_id, func.count())
        .where(Ticket.status == "confirmed",
               Ticket.consumed_by_version.is_(None),
               Ticket.category_id.is_not(None))
        .group_by(Ticket.category_id)
    ).all()
    return {category_id: count for category_id, count in rows}


def _out(db, category, classifier):
    counts = _available_counts(db)
    return category_out(
        category, classifier.labels, counts.get(category.id, 0),
        settings.min_tickets_new_category)


def _get(db, category_id):
    category = db.get(Category, category_id)
    if category is None:
        raise HTTPException(404, "Category not found.")
    return category


def _find_by_name(db, name):
    return db.scalar(select(Category).where(
        func.lower(Category.name) == name.lower()))


@router.get("", response_model=list[CategoryOut])
def list_categories(status: Optional[str] = None,
                    db: Session = Depends(get_db),
                    classifier=Depends(get_classifier)):
    query = select(Category).order_by(Category.created_at.desc())
    if status is not None:
        if status not in CATEGORY_STATUSES:
            raise HTTPException(
                422, f"status must be one of {sorted(CATEGORY_STATUSES)}")
        query = query.where(Category.status == status)

    counts = _available_counts(db)
    labels = classifier.labels
    minimum = settings.min_tickets_new_category
    return [category_out(c, labels, counts.get(c.id, 0), minimum)
            for c in db.scalars(query)]


@router.post("", response_model=CategoryOut, status_code=201)
def create_category(body: CategoryCreate, db: Session = Depends(get_db),
                    classifier=Depends(get_classifier)):
    """An admin creates a category directly (it starts approved)."""
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "Category name cannot be empty.")
    if _find_by_name(db, name) is not None:
        raise HTTPException(409, f"A category named '{name}' already exists.")

    category = Category(name=name, description=body.description,
                        status="approved")
    db.add(category)
    db.commit()
    db.refresh(category)
    return _out(db, category, classifier)


@router.patch("/{category_id}", response_model=CategoryOut)
def update_category(category_id: int, body: CategoryUpdate,
                    db: Session = Depends(get_db),
                    classifier=Depends(get_classifier)):
    """Only the description can change; the name is the model's label."""
    category = _get(db, category_id)
    category.description = body.description
    db.commit()
    return _out(db, category, classifier)


@router.post("/{category_id}/approve", response_model=CategoryOut)
def approve_category(category_id: int, db: Session = Depends(get_db),
                     classifier=Depends(get_classifier)):
    category = _get(db, category_id)
    if category.status != "proposed":
        raise HTTPException(
            409, f"Only proposed categories can be approved "
            f"(this one is {category.status}).")
    category.status = "approved"
    db.commit()
    return _out(db, category, classifier)


@router.post("/{category_id}/reject", response_model=CategoryOut)
def reject_category(category_id: int, db: Session = Depends(get_db),
                    classifier=Depends(get_classifier)):
    """Reject a proposal; its tickets go back to the review queue."""
    category = _get(db, category_id)
    if category.status != "proposed":
        raise HTTPException(
            409, f"Only proposed categories can be rejected "
            f"(this one is {category.status}).")

    db.execute(
        update(Ticket)
        .where(Ticket.category_id == category.id)
        .values(category_id=None, status="needs_review", reviewed_at=None))
    category.status = "rejected"
    db.commit()
    return _out(db, category, classifier)


@router.post("/{category_id}/merge", response_model=CategoryOut)
def merge_category(category_id: int, body: MergeRequest,
                   db: Session = Depends(get_db),
                   classifier=Depends(get_classifier)):
    """Fold a duplicate category into another approved one."""
    source = _get(db, category_id)
    target = _get(db, body.into_id)

    if source.id == target.id:
        raise HTTPException(422, "A category cannot be merged into itself.")
    if source.status not in ("proposed", "approved"):
        raise HTTPException(409, f"This category is already {source.status}.")
    if target.status != "approved":
        raise HTTPException(409, "The target category must be approved.")
    if source.name in classifier.labels:
        raise HTTPException(
            409, "This category is already part of the model; merging "
                 "trained categories is not supported yet.")

    db.execute(
        update(Ticket)
        .where(Ticket.category_id == source.id)
        .values(category_id=target.id))
    source.status = "merged"
    source.merged_into_id = target.id
    db.commit()
    return _out(db, source, classifier)
