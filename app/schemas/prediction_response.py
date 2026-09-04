from pydantic import BaseModel


class PredictionResponse(BaseModel):
    model_name: str
    prediction: float
    demand_category: str

