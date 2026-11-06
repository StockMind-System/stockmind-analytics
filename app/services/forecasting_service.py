from __future__ import annotations

import hashlib
import math
from datetime import timedelta

import pandas as pd
from xgboost import XGBRegressor

from app.ml.metrics import regression_metrics
from app.schemas.prediction_request import PredictionRequest


MODEL_VERSION = "weekly-demand-v1"
MINIMUM_XGBOOST_WEEKS = 12


def _normalise_history(payload: PredictionRequest) -> pd.Series:
    frame = pd.DataFrame(
        {
            "week_start": [item.week_start for item in payload.history],
            "demand": [item.demand for item in payload.history],
        }
    )
    frame["week_start"] = pd.to_datetime(frame["week_start"])
    frame = frame.groupby("week_start", as_index=False)["demand"].sum().sort_values("week_start")
    if frame.empty:
        raise ValueError("El historial semanal es obligatorio.")
    full_index = pd.date_range(
        start=frame["week_start"].min(), end=frame["week_start"].max(), freq="7D"
    )
    series = frame.set_index("week_start")["demand"].reindex(full_index, fill_value=0.0).astype(float)
    if len(series) < 4:
        raise ValueError("Se requieren al menos cuatro semanas de historial.")
    if not all(math.isfinite(value) and value >= 0 for value in series):
        raise ValueError("La demanda debe contener valores finitos y no negativos.")
    return series


def _feature_row(values: list[float], step: int) -> list[float]:
    recent = values[-4:]
    angle = 2 * math.pi * (step % 52) / 52
    return [recent[-1], recent[-2], recent[-3], recent[-4], sum(recent) / 4, math.sin(angle), math.cos(angle)]


def _training_matrix(values: list[float]) -> tuple[pd.DataFrame, pd.Series]:
    rows: list[list[float]] = []
    targets: list[float] = []
    for index in range(4, len(values)):
        rows.append(_feature_row(values[:index], index))
        targets.append(values[index])
    columns = ["lag_1", "lag_2", "lag_3", "lag_4", "moving_average_4", "week_sin", "week_cos"]
    return pd.DataFrame(rows, columns=columns), pd.Series(targets, dtype=float)


def _moving_average_forecast(values: list[float], horizon: int) -> list[float]:
    working = list(values)
    forecasts: list[float] = []
    for _ in range(horizon):
        value = max(0.0, sum(working[-4:]) / min(4, len(working)))
        forecasts.append(value)
        working.append(value)
    return forecasts


def _xgboost_forecast(values: list[float], horizon: int) -> list[float]:
    features, target = _training_matrix(values)
    model = XGBRegressor(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )
    model.fit(features, target)
    working = list(values)
    forecasts: list[float] = []
    for _ in range(horizon):
        row = pd.DataFrame([_feature_row(working, len(working))], columns=features.columns)
        value = max(0.0, float(model.predict(row)[0]))
        forecasts.append(value)
        working.append(value)
    return forecasts


def _select_method(values: list[float], horizon: int) -> str:
    if len(values) < MINIMUM_XGBOOST_WEEKS:
        return "moving_average_4"
    holdout = min(max(2, horizon), 4, len(values) - 8)
    train = values[:-holdout]
    actual = values[-holdout:]
    baseline = _moving_average_forecast(train, holdout)
    candidate = _xgboost_forecast(train, holdout)
    baseline_metrics = regression_metrics(actual, baseline)
    candidate_metrics = regression_metrics(actual, candidate)
    baseline_score = baseline_metrics["wape"] if baseline_metrics["wape"] is not None else baseline_metrics["mae"]
    candidate_score = candidate_metrics["wape"] if candidate_metrics["wape"] is not None else candidate_metrics["mae"]
    return "xgboost" if float(candidate_score) < float(baseline_score) else "moving_average_4"


def create_forecast(payload: PredictionRequest) -> dict[str, object]:
    series = _normalise_history(payload)
    values = series.tolist()
    method = _select_method(values, payload.horizon_weeks)
    forecast_values = (
        _xgboost_forecast(values, payload.horizon_weeks)
        if method == "xgboost"
        else _moving_average_forecast(values, payload.horizon_weeks)
    )
    canonical = "|".join(f"{index.date().isoformat()}:{value:.6f}" for index, value in series.items())
    dataset_version = f"sha256-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:12]}"
    cutoff = series.index.max().date()
    return {
        "sku": payload.sku.strip().upper(),
        "cutoff_date": cutoff,
        "horizon_weeks": payload.horizon_weeks,
        "unit": payload.unit.strip(),
        "method": method,
        "model_version": f"{payload.model_name}-{MODEL_VERSION}",
        "dataset_version": dataset_version,
        "status": "valid",
        "forecasts": [
            {
                "week_start": cutoff + timedelta(days=7 * (index + 1)),
                "predicted_demand": round(value, 4),
            }
            for index, value in enumerate(forecast_values)
        ],
    }
