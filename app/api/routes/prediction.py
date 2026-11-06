from fastapi import APIRouter, HTTPException

from app.schemas.prediction_request import PredictionRequest
from app.schemas.prediction_response import PredictionResponse
from app.services.forecasting_service import create_forecast


router = APIRouter(prefix="/prediction", tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
def create_prediction(payload: PredictionRequest) -> PredictionResponse:
    try:
        return PredictionResponse(**create_forecast(payload))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

