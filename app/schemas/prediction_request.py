from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "demand_model",
                "features": {"price": 19.9, "stock": 80, "promotion": 1},
            }
        }
    )

    model_name: str = Field(default="demand_model", pattern=r"^[A-Za-z0-9_-]+$")
    features: dict[str, float | int | str] = Field(min_length=1)

