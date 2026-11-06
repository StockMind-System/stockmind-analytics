from app.ml.demand_classifier import classify_demand
from app.ml.metrics import regression_metrics


def test_demand_categories() -> None:
    assert classify_demand(20) == "low"
    assert classify_demand(50) == "medium"
    assert classify_demand(80) == "high"


def test_perfect_regression_metrics() -> None:
    result = regression_metrics([1, 2, 3], [1, 2, 3])

    assert result == {
        "mae": 0.0,
        "rmse": 0.0,
        "wape": 0.0,
        "r2": 1.0,
        "reason": None,
    }


def test_wape_is_not_calculable_when_actual_total_is_zero() -> None:
    result = regression_metrics([0, 0], [1, 0])

    assert result["wape"] is None
    assert result["reason"] is not None

