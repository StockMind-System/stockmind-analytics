from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBRegressor


MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str,
) -> Path:
    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )
    model.fit(features, target)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(
        {"model": model, "feature_columns": list(features.columns)}, model_path
    )
    return model_path

