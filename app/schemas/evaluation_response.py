from pydantic import BaseModel


class EvaluationResponse(BaseModel):
    mae: float
    rmse: float
    r2: float

