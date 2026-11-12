"""Turn database rows into API response objects."""

from datetime import timezone

from backend.app.schemas import (
    CategoryOut,
    CategoryRef,
    JobOut,
    ModelOut,
    PredictionItem,
    TicketOut,
)


def utc(value):
    """Mark a naive UTC timestamp as UTC so clients parse it correctly."""
    return value.replace(tzinfo=timezone.utc) if value is not None else None


def ticket_out(ticket):
    category = ticket.category
    return TicketOut(
        id=ticket.id,
        text=ticket.text,
        status=ticket.status,
        predicted_label=ticket.predicted_label,
        confidence=ticket.confidence,
        top_predictions=[
            PredictionItem(**item) for item in (ticket.top_predictions or [])],
        is_spot_check=ticket.is_spot_check,
        category=(CategoryRef(id=category.id, name=category.name,
                              status=category.status) if category else None),
        model_version=ticket.model_version,
        created_at=utc(ticket.created_at),
        reviewed_at=utc(ticket.reviewed_at),
    )


def category_out(category, model_labels, available, min_new):
    in_model = category.name in model_labels
    needed = None
    if category.status == "approved" and not in_model:
        needed = max(0, min_new - available)

    return CategoryOut(
        id=category.id,
        name=category.name,
        description=category.description,
        status=category.status,
        merged_into_id=category.merged_into_id,
        created_at=utc(category.created_at),
        in_model=in_model,
        tickets_available=available,
        tickets_needed=needed,
    )


def job_summary(report):
    if not report:
        return None
    return {key: report.get(key) for key in (
        "new_categories", "gate", "accuracy", "old_accuracy",
        "old_accuracy_drop", "new_category_accuracy")}


def job_out(job, include_report=False):
    return JobOut(
        id=job.id,
        status=job.status,
        created_at=utc(job.created_at),
        started_at=utc(job.started_at),
        finished_at=utc(job.finished_at),
        base_version=job.base_version,
        version=job.version,
        error=job.error,
        n_tickets=len(job.ticket_ids or []),
        summary=job_summary(job.report),
        report=job.report if include_report else None,
    )


def model_out(registry, name):
    manifest = registry.read_manifest(name)
    metrics = registry.read_metrics(name)
    calibration = registry.read_calibration(name)
    categories = metrics.get("categories")

    return ModelOut(
        name=name,
        parent=manifest.get("parent"),
        created_at=manifest.get("created_at"),
        notes=manifest.get("notes", ""),
        active=registry.active() == name,
        complete=registry.is_complete(name),
        n_categories=len(categories) if categories else None,
        new_categories=metrics.get("new_categories", []),
        accuracy=metrics.get("accuracy"),
        old_accuracy=metrics.get("old_accuracy"),
        old_accuracy_drop=metrics.get("old_accuracy_drop"),
        new_category_accuracy=metrics.get("new_category_accuracy"),
        gate=metrics.get("gate"),
        calibration=calibration,
    )
