"""FastAPI dependencies: database session and the shared services."""

from fastapi import Request

from backend.app.database import SessionLocal


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_registry(request: Request):
    return request.app.state.registry


def get_classifier(request: Request):
    return request.app.state.classifier


def get_jobs(request: Request):
    return request.app.state.jobs
