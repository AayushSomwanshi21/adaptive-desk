"""Database tables: categories, tickets and training jobs."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base


def utcnow():
    """Naive UTC timestamp (SQLite drops time zones); serialised as UTC."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, default="")

    # proposed -> approved (or rejected / merged into another category).
    # Only approved categories are used for training.
    status: Mapped[str] = mapped_column(String(20), default="approved",
                                        index=True)
    merged_into_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("categories.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text)

    # needs_review: waiting for a human
    # auto_labeled: confident prediction, accepted automatically
    # confirmed:    a human chose the category (used for training)
    status: Mapped[str] = mapped_column(String(20), index=True,
                                        default="needs_review")

    # What the model said when the ticket arrived.
    predicted_label: Mapped[Optional[str]] = mapped_column(
        String(120), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    top_predictions: Mapped[list] = mapped_column(JSON, default=list)
    model_version: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True)
    is_spot_check: Mapped[bool] = mapped_column(Boolean, default=False)

    # The final label (set when auto-labelled or confirmed by a human).
    category_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("categories.id"), nullable=True, index=True)
    category: Mapped[Optional[Category]] = relationship(
        foreign_keys=[category_id])

    # The model version whose training run already used (or held out) this
    # ticket. Set when that version is promoted.
    consumed_by_version: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True)


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # queued -> running -> succeeded | failed
    status: Mapped[str] = mapped_column(String(20), default="queued",
                                        index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True)

    base_version: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True)
    version: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    ticket_ids: Mapped[list] = mapped_column(JSON, default=list)
    report: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
