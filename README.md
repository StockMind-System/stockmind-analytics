# StockMind Analytics

Servicio FastAPI para pronóstico semanal de demanda. Recibe desde la API C#
un historial semanal por SKU, compara XGBoost con un promedio móvil de cuatro
semanas mediante una reserva temporal y devuelve un horizonte configurable
(cuatro semanas por defecto). También calcula MAE, RMSE, WAPE y R².

## Ejecución

```powershell
uv sync --dev
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

La documentación OpenAPI queda disponible en `http://localhost:8000/docs` y
la salud en `GET /health`.

## Pruebas

```powershell
uv run pytest -q
```

Los datos del piloto son sintéticos. El servicio respeta el orden temporal:
las semanas reservadas para evaluación no participan en el entrenamiento que
las predice. Si el historial todavía no alcanza 12 semanas, utiliza el
promedio móvil como contingencia identificada.
