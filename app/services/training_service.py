from collections.abc import Mapping, Sequence

import pandas as pd

from app.ml.model_trainer import train_model
from app.services.preprocessing_service import preprocess_records


def train_demand_model(
    records: Sequence[Mapping[str, object]],
    target_column: str,
    model_name: str,
) -> dict[str, object]:
    frame = pd.DataFrame(records)
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' is missing")
    if len(frame) < 2:
        raise ValueError("At least two training records are required")

    target = pd.to_numeric(frame.pop(target_column), errors="raise")
    features = preprocess_records(frame.to_dict(orient="records"))
    model_path = train_model(features, target, model_name)
    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "training_rows": len(frame),
        "feature_count": len(features.columns),
    }

