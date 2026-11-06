from datetime import date

from pydantic import BaseModel


class WeeklyForecastResponse(BaseModel):
    week_start: date
    predicted_demand: float


class PredictionResponse(BaseModel):
    sku: str
    cutoff_date: date
    horizon_weeks: int
    unit: str
    method: str
    model_version: str
    dataset_version: str
    status: str
    forecasts: list[WeeklyForecastResponse]

