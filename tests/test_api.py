from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_evaluation() -> None:
    response = client.post(
        "/api/v1/evaluation/evaluate",
        json={"y_true": [10, 20, 30], "y_pred": [12, 18, 30]},
    )

    assert response.status_code == 200
    assert response.json()["mae"] == 4 / 3
    assert response.json()["wape"] == 100 * 4 / 60


def test_prediction_accepts_backend_weekly_contract() -> None:
    response = client.post(
        "/api/v1/prediction/predict",
        json={
            "sku": "SKU-001",
            "model_name": "demand_model",
            "horizon_weeks": 4,
            "unit": "units",
            "history": [
                {"week_start": "2026-09-07", "demand": 10},
                {"week_start": "2026-09-14", "demand": 12},
                {"week_start": "2026-09-21", "demand": 9},
                {"week_start": "2026-09-28", "demand": 13},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sku"] == "SKU-001"
    assert body["horizon_weeks"] == 4
    assert body["method"] == "moving_average_4"
    assert len(body["forecasts"]) == 4

