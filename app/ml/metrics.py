import math

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(
    y_true: list[float], y_pred: list[float]
) -> dict[str, float | str | None]:
    absolute_total = sum(abs(value) for value in y_true)
    wape = (
        100.0 * sum(abs(actual - predicted) for actual, predicted in zip(y_true, y_pred)) / absolute_total
        if absolute_total > 0
        else None
    )
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None
    if r2 is not None and not math.isfinite(r2):
        r2 = None
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "wape": wape,
        "r2": r2,
        "reason": None if wape is not None else "WAPE no calculable: la demanda real total es cero.",
    }

