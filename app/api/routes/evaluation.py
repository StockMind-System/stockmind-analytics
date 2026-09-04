from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.schemas.evaluation_response import EvaluationResponse
from app.services.evaluation_service import evaluate_predictions


router = APIRouter(prefix="/evaluation", tags=["evaluation"])


class EvaluationRequest(BaseModel):
    y_true: list[float] = Field(min_length=2)
    y_pred: list[float] = Field(min_length=2)


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate(payload: EvaluationRequest) -> EvaluationResponse:
    try:
        return EvaluationResponse(
            **evaluate_predictions(payload.y_true, payload.y_pred)
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

