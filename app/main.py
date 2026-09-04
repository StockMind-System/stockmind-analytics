from fastapi import FastAPI

from app.api.routes import evaluation, prediction, training


app = FastAPI(
    title="StockMind Analytics API",
    version="0.1.0",
    description="Training, prediction and evaluation services for product demand.",
)

app.include_router(prediction.router, prefix="/api/v1")
app.include_router(training.router, prefix="/api/v1")
app.include_router(evaluation.router, prefix="/api/v1")


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}

