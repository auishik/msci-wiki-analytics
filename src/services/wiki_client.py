import logging
import random

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import Settings, get_settings
from src.exceptions import RecoverableAPIError, WikipediaAPIError

logger = logging.getLogger(__name__)

BASE_URL = "https://en.wikipedia.org/w/api.php"


class RetryAfterWait:
    """Custom wait strategy that uses Retry-After header if available."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._exponential = wait_exponential(
            multiplier=settings.retry_multiplier,
            min=settings.retry_min_wait,
            max=settings.retry_max_wait,
        )

    def __call__(self, retry_state: RetryCallState) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exc, RecoverableAPIError) and exc.retry_after is not None:
            jitter = random.uniform(0, self._settings.retry_jitter_max)
            return exc.retry_after + jitter
        wait_time: float = self._exponential(retry_state)
        return wait_time


class WikipediaAPIClient:
    """HTTP client for Wikipedia API with retry and rate limiting."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "User-Agent": self._settings.wiki_user_agent,
                    "Api-User-Agent": self._settings.wiki_user_agent,
                },
                timeout=httpx.Timeout(
                    connect=self._settings.timeout_connect,
                    read=self._settings.timeout_read,
                    write=self._settings.timeout_write,
                    pool=self._settings.timeout_pool,
                ),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get(self, params: dict[str, str]) -> httpx.Response:
        """Make a GET request to the Wikipedia API with retry logic."""
        retrying = retry(
            retry=retry_if_exception_type(RecoverableAPIError),
            stop=stop_after_attempt(self._settings.retry_max_attempts),
            wait=RetryAfterWait(self._settings),
            reraise=True,
        )

        @retrying
        async def _do_request() -> httpx.Response:
            return await self._make_request(params)

        return await _do_request()

    async def _make_request(self, params: dict[str, str]) -> httpx.Response:
        """Make a single HTTP request, raising appropriate exceptions."""
        client = await self._get_client()
        context = params.get("page", "unknown")

        try:
            response = await client.get(BASE_URL, params=params)
        except httpx.TimeoutException as e:
            logger.warning("Timeout for '%s': %s", context, e)
            raise RecoverableAPIError(f"Timeout for '{context}'", cause=e) from e
        except httpx.ConnectError as e:
            logger.warning("Connection error for '%s': %s", context, e)
            raise RecoverableAPIError(
                f"Connection failed for '{context}'", cause=e
            ) from e
        except httpx.RequestError as e:
            logger.error("Request error for '%s': %s", context, e)
            raise WikipediaAPIError(f"Request failed for '{context}'", cause=e) from e

        if response.status_code == 429:
            retry_after = self._parse_retry_after(response)
            logger.warning(
                "Rate limited for '%s', retry-after: %s", context, retry_after
            )
            raise RecoverableAPIError(
                f"Rate limited for '{context}'",
                status_code=429,
                retry_after=retry_after,
            )

        if response.status_code >= 500:
            logger.warning("Server error %d for '%s'", response.status_code, context)
            raise RecoverableAPIError(
                f"Server error {response.status_code} for '{context}'",
                status_code=response.status_code,
            )

        if response.status_code >= 400:
            logger.error("HTTP error %d for '%s'", response.status_code, context)
            raise WikipediaAPIError(
                f"HTTP {response.status_code} for '{context}'",
                status_code=response.status_code,
            )

        return response

    def _parse_retry_after(self, response: httpx.Response) -> float | None:
        """Parse Retry-After header value."""
        retry_after = response.headers.get("Retry-After")
        if retry_after is None:
            return None
        try:
            return float(retry_after)
        except ValueError:
            logger.warning("Could not parse Retry-After header: %s", retry_after)
            return None
