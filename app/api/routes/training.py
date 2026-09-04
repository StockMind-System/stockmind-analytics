from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.training_service import train_demand_model


router = APIRouter(prefix="/training", tags=["training"])


class TrainingRequest(BaseModel):
    model_name: str = Field(default="demand_model", pattern=r"^[A-Za-z0-9_-]+$")
    target_column: str
    records: list[dict[str, Any]] = Field(min_length=2)


class TrainingResponse(BaseModel):
    model_name: str
    model_path: str
    training_rows: int
    feature_count: int


@router.post("/train", response_model=TrainingResponse)
def train(payload: TrainingRequest) -> TrainingResponse:
    try:
        result = train_demand_model(
            payload.records, payload.target_column, payload.model_name
        )
        return TrainingResponse(**result)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

