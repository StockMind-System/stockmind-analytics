from pydantic import BaseModel


class EvaluationResponse(BaseModel):
    mae: float
    rmse: float
    wape: float | None
    r2: float | None
    reason: str | None = None

