"""AdaptiveDesk API.

Run from the repo root:

    uvicorn backend.app.main:app --reload

Interactive docs: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.classifier import ClassifierService
from backend.app.config import settings
from backend.app.database import SessionLocal, init_db
from backend.app.routers import categories, models, stats, tickets, training
from backend.app.services import JobRunner, sync_categories
from ml.serving.versioning import ModelRegistry


@asynccontextmanager
async def lifespan(app):
    init_db()

    registry = ModelRegistry(settings.registry_root)
    classifier = ClassifierService(registry, settings.model_name)
    classifier.load_active()           # no-op until a model has been promoted

    with SessionLocal() as db:
        sync_categories(db, classifier.labels)

    jobs = JobRunner(SessionLocal, registry, classifier)
    jobs.mark_interrupted()

    app.state.registry = registry
    app.state.classifier = classifier
    app.state.jobs = jobs
    yield


app = FastAPI(title="AdaptiveDesk", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tickets.router)
app.include_router(categories.router)
app.include_router(training.router)
app.include_router(models.router)
app.include_router(stats.router)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "model": app.state.classifier.info()}
