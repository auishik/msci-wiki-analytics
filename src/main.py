from fastapi import FastAPI

from src.models.schemas import HealthResponse

app = FastAPI(
    title="Wikipedia Word-Frequency API",
    description="API for generating word-frequency dictionaries from Wikipedia",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="healthy")
