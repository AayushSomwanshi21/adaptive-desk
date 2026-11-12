"""Model versions: list, inspect, promote, roll back."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.deps import get_classifier, get_db, get_registry
from backend.app.schemas import ModelOut, PromoteRequest
from backend.app.serializers import model_out
from backend.app.services import mark_consumed, sync_categories

router = APIRouter(prefix="/models", tags=["models"])


def _require_version(registry, name):
    if name not in registry.list_versions():
        raise HTTPException(404, f"Unknown model version: {name}")


@router.get("", response_model=list[ModelOut])
def list_models(registry=Depends(get_registry)):
    return [model_out(registry, name)
            for name in sorted(registry.list_versions(), reverse=True)]


@router.get("/active", response_model=ModelOut)
def get_active(registry=Depends(get_registry)):
    active = registry.active()
    if active is None:
        raise HTTPException(404, "No model is active yet.")
    return model_out(registry, active)


@router.post("/rollback", response_model=ModelOut)
def rollback(registry=Depends(get_registry),
             classifier=Depends(get_classifier)):
    """Go back to the previously promoted version.

    Tickets already marked as used by the rolled-back version stay marked.
    """
    try:
        active = registry.rollback()
    except RuntimeError as error:
        raise HTTPException(409, str(error))

    classifier.load_active()
    return model_out(registry, active)


@router.get("/{name}", response_model=ModelOut)
def get_model(name: str, registry=Depends(get_registry)):
    _require_version(registry, name)
    return model_out(registry, name)


@router.post("/{name}/promote", response_model=ModelOut)
def promote(name: str, body: PromoteRequest = PromoteRequest(),
            db: Session = Depends(get_db), registry=Depends(get_registry),
            classifier=Depends(get_classifier)):
    """Make a version the one that serves tickets.

    A version whose promotion gate failed is refused unless force is set.
    """
    _require_version(registry, name)
    if not registry.is_complete(name):
        raise HTTPException(409, f"Version {name} is incomplete.")

    gate = registry.read_metrics(name).get("gate")
    if gate is not None and not gate.get("passed") and not body.force:
        failed = [c["detail"] for c in gate.get("checks", [])
                  if not c.get("passed")]
        raise HTTPException(409, {
            "message": "The promotion gate failed for this version.",
            "failed_checks": failed,
            "hint": "Send {\"force\": true} to promote it anyway.",
        })

    previous = registry.active()
    registry.promote(name)
    try:
        classifier.load_active()
    except Exception as error:
        # Do not leave the registry pointing at a model that cannot be loaded.
        if previous is not None:
            registry.rollback()
        raise HTTPException(500, f"Could not load {name}: {error}")

    sync_categories(db, classifier.labels)
    mark_consumed(db, name)
    return model_out(registry, name)
