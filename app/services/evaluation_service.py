from app.ml.metrics import regression_metrics


def evaluate_predictions(
    y_true: list[float], y_pred: list[float]
) -> dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) < 2:
        raise ValueError("At least two observations are required")
    return regression_metrics(y_true, y_pred)

