"""SQLAlchemy engine, session factory and table creation."""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.app.config import settings


class Base(DeclarativeBase):
    pass


def _make_engine(url):
    if url.startswith("sqlite"):
        # Make sure the folder of the database file exists.
        path = url.replace("sqlite:///", "", 1)
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # The training job runs in a thread with its own session.
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url)


engine = _make_engine(settings.database_url)
SessionLocal = sessionmaker(
    bind=engine, autoflush=False, expire_on_commit=False)


def init_db():
    from backend.app import models  # noqa: F401  (registers the tables)
    Base.metadata.create_all(engine)
