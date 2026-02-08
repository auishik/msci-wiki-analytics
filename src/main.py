import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from src.exceptions import WikipediaAPIError, WikipediaParseError
from src.models.schemas import HealthResponse, KeywordsRequest, WordFrequencyResponse
from src.services.frequency import WordFrequencyService
from src.services.wiki_client import WikipediaAPIClient
from src.services.wikipedia import WikiRecursiveFetchService

logger = logging.getLogger(__name__)

# Service instances
api_client: WikipediaAPIClient | None = None
wiki_service: WikiRecursiveFetchService | None = None
frequency_service: WordFrequencyService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan - startup and shutdown."""
    global api_client, wiki_service, frequency_service
    api_client = WikipediaAPIClient()
    wiki_service = WikiRecursiveFetchService(api_client=api_client)
    frequency_service = WordFrequencyService()
    logger.info("Services initialized")
    yield
    if api_client:
        await api_client.close()
    logger.info("Services shut down")


app = FastAPI(
    title="Wikipedia Word-Frequency API",
    description="API for generating word-frequency dictionaries from Wikipedia",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(WikipediaAPIError)
async def wikipedia_api_error_handler(
    request: Request, exc: WikipediaAPIError
) -> JSONResponse:
    """Handle Wikipedia API errors."""
    logger.error("Wikipedia API error: %s", exc)
    status_code = exc.status_code if exc.status_code else 502
    return JSONResponse(
        status_code=status_code,
        content={"detail": str(exc)},
    )


@app.exception_handler(WikipediaParseError)
async def wikipedia_parse_error_handler(
    request: Request, exc: WikipediaParseError
) -> JSONResponse:
    """Handle Wikipedia parse errors."""
    logger.error("Wikipedia parse error: %s", exc)
    return JSONResponse(
        status_code=502,
        content={"detail": str(exc)},
    )


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Check if the service is healthy."""
    return HealthResponse(status="healthy")


@app.get("/word-frequency", response_model=WordFrequencyResponse)
async def get_word_frequency(
    article: str = Query(description="The title of the Wikipedia article"),
    depth: int = Query(ge=0, description="The depth of traversal"),
) -> WordFrequencyResponse:
    """
    Get word frequency dictionary for a Wikipedia article.

    Traverses the article and its linked articles up to the specified depth,
    and returns word counts and percentages.
    """
    if wiki_service is None or frequency_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    result = await wiki_service.traverse(article, depth)

    if not result.texts:
        raise HTTPException(status_code=404, detail=f"Article '{article}' not found")

    return frequency_service.calculate(result.texts)


@app.post("/keywords", response_model=WordFrequencyResponse)
async def get_keywords(request: KeywordsRequest) -> WordFrequencyResponse:
    """
    Get filtered word frequency dictionary for a Wikipedia article.

    Similar to /word-frequency, but filters out words in the ignore list
    and applies a percentile threshold.
    """
    if wiki_service is None or frequency_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    result = await wiki_service.traverse(request.article, request.depth)

    if not result.texts:
        raise HTTPException(
            status_code=404, detail=f"Article '{request.article}' not found"
        )

    return frequency_service.calculate(
        texts=result.texts,
        ignore_list=request.ignore_list,
        percentile=request.percentile,
    )
