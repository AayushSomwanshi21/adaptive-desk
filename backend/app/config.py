"""Backend settings, read from environment variables with sensible defaults."""

import math
import os
from dataclasses import dataclass
from pathlib import Path

from ml.continual_learning.promotion import GateSettings

ROOT = Path(__file__).resolve().parents[2]  # the repo root


def _default_database_url():
    path = ROOT / "backend" / "data" / "adaptivedesk.db"
    return f"sqlite:///{path.as_posix()}"


@dataclass(frozen=True)
class Settings:
    database_url: str = os.environ.get(
        "ADAPTIVEDESK_DB", _default_database_url())

    # Where model versions live (see ml/serving/versioning.py).
    registry_root: Path = Path(os.environ.get(
        "ADAPTIVEDESK_REGISTRY", ROOT / "ml" / "models" / "registry"))

    model_name: str = os.environ.get(
        "ADAPTIVEDESK_MODEL_NAME", "distilbert-base-uncased")

    # How many guesses to show agents.
    top_k: int = 3

    # Share of confidently auto-labelled tickets that are still sent to a
    # human, to notice when the model is confidently wrong.
    spot_check_rate: float = float(
        os.environ.get("ADAPTIVEDESK_SPOT_CHECK_RATE", "0.05"))

    # Must match what the training job uses (see services.py).
    holdout_fraction: float = 0.25

    # Confirmed tickets of a category the model already knows that trigger a
    # retraining run on their own.
    min_tickets_existing_category: int = 30

    cors_origins: tuple = ("http://localhost:5173", "http://127.0.0.1:5173")

    @property
    def min_tickets_new_category(self):
        """Confirmed tickets a NEW category needs before it can be trained.

        The training run holds out `holdout_fraction` of them and the promotion
        gate needs `min_new_eval_examples` held-out tickets to judge it.
        """
        needed = GateSettings().min_new_eval_examples
        return math.ceil(needed / self.holdout_fraction)


settings = Settings()
