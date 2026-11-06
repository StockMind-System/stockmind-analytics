from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class WeeklyDemandInput(BaseModel):
    week_start: date
    demand: float = Field(ge=0)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sku": "HE-MAR-016",
                "model_name": "demand_model",
                "horizon_weeks": 4,
                "unit": "units",
                "history": [
                    {"week_start": "2026-09-07", "demand": 18},
                    {"week_start": "2026-09-14", "demand": 21},
                    {"week_start": "2026-09-21", "demand": 19},
                    {"week_start": "2026-09-28", "demand": 24},
                ],
            }
        }
    )

    sku: str = Field(min_length=1, max_length=80)
    model_name: str = Field(default="demand_model", pattern=r"^[A-Za-z0-9_-]+$")
    horizon_weeks: int = Field(default=4, ge=1, le=12)
    unit: str = Field(default="units", min_length=1, max_length=32)
    history: list[WeeklyDemandInput] = Field(min_length=4)

