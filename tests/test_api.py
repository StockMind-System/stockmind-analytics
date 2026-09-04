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


def test_prediction_requires_trained_model() -> None:
    response = client.post(
        "/api/v1/prediction/predict",
        json={"model_name": "missing", "features": {"price": 10}},
    )

    assert response.status_code == 404

