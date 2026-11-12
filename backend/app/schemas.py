"""Request and response shapes of the API."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ---- tickets ---------------------------------------------------------------
class PredictionItem(BaseModel):
    label: str
    confidence: float


class CategoryRef(BaseModel):
    id: int
    name: str
    status: str


class TicketCreate(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


class TicketOut(BaseModel):
    id: int
    text: str
    status: str
    predicted_label: Optional[str]
    confidence: Optional[float]
    top_predictions: list[PredictionItem]
    is_spot_check: bool
    category: Optional[CategoryRef]
    model_version: Optional[str]
    created_at: datetime
    reviewed_at: Optional[datetime]


class TicketList(BaseModel):
    items: list[TicketOut]
    total: int


class NewCategory(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""


class LabelRequest(BaseModel):
    """Pick an existing category OR propose a new one."""
    category_id: Optional[int] = None
    new_category: Optional[NewCategory] = None

    @model_validator(mode="after")
    def exactly_one(self):
        if (self.category_id is None) == (self.new_category is None):
            raise ValueError(
                "Give either category_id or new_category, not both.")
        return self


class PredictOut(BaseModel):
    model_version: Optional[str]
    label: Optional[str]
    confidence: Optional[float]
    top: list[PredictionItem]
    needs_review: bool


# ---- categories ------------------------------------------------------------
class CategoryCreate(NewCategory):
    pass


class CategoryUpdate(BaseModel):
    description: str


class MergeRequest(BaseModel):
    into_id: int


class CategoryOut(BaseModel):
    id: int
    name: str
    description: str
    status: str
    merged_into_id: Optional[int]
    created_at: datetime
    in_model: bool            # the active model already has this category
    tickets_available: int    # confirmed tickets not yet used for training
    tickets_needed: Optional[int]  # more needed before it can be trained


# ---- training --------------------------------------------------------------
class PlanCategory(BaseModel):
    name: str
    tickets: int
    needed: int


class TrainingPlan(BaseModel):
    can_start: bool
    reason: Optional[str]
    ready_categories: list[PlanCategory]
    collecting: list[PlanCategory]
    min_tickets_new_category: int
    n_tickets: int


class JobOut(BaseModel):
    id: int
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    base_version: Optional[str]
    version: Optional[str]
    error: Optional[str]
    n_tickets: int
    summary: Optional[dict]
    report: Optional[dict] = None


# ---- models ----------------------------------------------------------------
class PromoteRequest(BaseModel):
    force: bool = False   # promote even though the promotion gate failed


class ModelOut(BaseModel):
    name: str
    parent: Optional[str]
    created_at: Optional[str]
    notes: str
    active: bool
    complete: bool
    n_categories: Optional[int]
    new_categories: list[str]
    accuracy: Optional[float]
    old_accuracy: Optional[float]
    old_accuracy_drop: Optional[float]
    new_category_accuracy: Optional[float]
    gate: Optional[dict]
    calibration: dict
