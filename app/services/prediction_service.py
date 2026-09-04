from collections.abc import Mapping

from app.ml.demand_classifier import classify_demand
from app.ml.model_predictor import load_model_bundle, predict
from app.services.preprocessing_service import preprocess_records


def predict_demand(model_name: str, feature_values: Mapping[str, object]) -> dict:
    bundle = load_model_bundle(model_name)
    features = preprocess_records(
        [feature_values], expected_columns=bundle["feature_columns"]
    )
    value = predict(bundle, features)
    return {
        "model_name": model_name,
        "prediction": value,
        "demand_category": classify_demand(value),
    }

