from pathlib import Path

import joblib
import pandas as pd

from app.ml.model_trainer import MODELS_DIR


def load_model_bundle(model_name: str) -> dict:
    model_path = MODELS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model '{model_name}' was not found")
    return joblib.load(model_path)


def predict(bundle: dict, features: pd.DataFrame) -> float:
    return float(bundle["model"].predict(features)[0])

