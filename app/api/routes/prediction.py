from fastapi import APIRouter, HTTPException

from app.schemas.prediction_request import PredictionRequest
from app.schemas.prediction_response import PredictionResponse
from app.services.prediction_service import predict_demand


router = APIRouter(prefix="/prediction", tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
def create_prediction(payload: PredictionRequest) -> PredictionResponse:
    try:
        return PredictionResponse(**predict_demand(payload.model_name, payload.features))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

