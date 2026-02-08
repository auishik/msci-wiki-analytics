import asyncio
import json
import logging
import random
import re
from functools import partial
from urllib.parse import unquote

import aiometer
import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import Settings, get_settings
from src.exceptions import (
    RecoverableAPIError,
    WikipediaAPIError,
    WikipediaFetchError,
    WikipediaParseError,
)

logger = logging.getLogger(__name__)


class PageContent(BaseModel):
    """Content extracted from a Wikipedia page."""

    title: str
    text: str
    links: list[str]


class TraversalError(BaseModel):
    """Error encountered while traversing a Wikipedia article."""

    title: str
    error: str


class TraversalResult(BaseModel):
    """Result of traversing Wikipedia articles."""

    texts: list[str] = Field(default_factory=list)
    visited: set[str] = Field(default_factory=set)
    errors: list[TraversalError] = Field(default_factory=list)


def _create_retry_wait(settings: Settings) -> "RetryAfterWait":
    """Create a wait strategy that respects Retry-After header."""
    return RetryAfterWait(settings)


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


class WikiRecursiveFetchService:
    """Service for fetching and recursively traversing Wikipedia articles."""

    BASE_URL = "https://en.wikipedia.org/w/api.php"

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

    async def fetch_page(self, title: str) -> PageContent | None:
        """
        Fetch a Wikipedia page's content and links.

        Args:
            title: The title of the Wikipedia article.

        Returns:
            PageContent with text and links, or None if page not found.

        Raises:
            WikipediaAPIError: If the API request fails after retries.
            WikipediaParseError: If unable to parse the response.
        """
        logger.debug("Fetching page: %s", title)

        params = {
            "action": "parse",
            "page": title,
            "format": "json",
            "prop": "text|links",
            "redirects": "1",
        }

        response = await self._fetch_with_retry(title, params)

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response for page '%s': %s", title, e)
            raise WikipediaParseError(
                f"Invalid JSON response for '{title}'", cause=e
            ) from e

        if "error" in data:
            logger.warning("Page not found: %s", title)
            return None

        parse_data = data.get("parse", {})
        html_content = parse_data.get("text", {}).get("*", "")

        # Extract plain text from HTML
        soup = BeautifulSoup(html_content, "lxml")

        # Remove unwanted elements
        for element in soup.find_all(["script", "style", "table", "sup"]):
            element.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = self._clean_text(text)

        # Extract internal Wikipedia links
        try:
            raw_links = parse_data.get("links", [])
            links = [
                link["*"]
                for link in raw_links
                if link.get("ns") == 0 and link.get("exists") is not None
            ]
        except KeyError as e:
            logger.error("Failed to parse links for page '%s': %s", title, e)
            raise WikipediaParseError(
                f"Failed to parse links for '{title}'", cause=e
            ) from e

        logger.debug(
            "Fetched page '%s': %d chars, %d links",
            title,
            len(text),
            len(links),
        )
        return PageContent(
            title=parse_data.get("title", title),
            text=text,
            links=links,
        )

    async def _fetch_with_retry(
        self, title: str, params: dict[str, str]
    ) -> httpx.Response:
        """Fetch a page with retry logic for recoverable errors."""
        retrying = retry(
            retry=retry_if_exception_type(RecoverableAPIError),
            stop=stop_after_attempt(self._settings.retry_max_attempts),
            wait=_create_retry_wait(self._settings),
            reraise=True,
        )

        @retrying
        async def _do_fetch() -> httpx.Response:
            return await self._make_request(title, params)

        return await _do_fetch()

    async def _make_request(self, title: str, params: dict[str, str]) -> httpx.Response:
        """Make a single HTTP request, raising appropriate exceptions."""
        client = await self._get_client()

        try:
            response = await client.get(self.BASE_URL, params=params)
        except httpx.TimeoutException as e:
            logger.warning("Timeout fetching page '%s': %s", title, e)
            raise RecoverableAPIError(
                f"Timeout fetching page '{title}'", cause=e
            ) from e
        except httpx.ConnectError as e:
            logger.warning("Connection error fetching page '%s': %s", title, e)
            raise RecoverableAPIError(
                f"Connection failed for '{title}'", cause=e
            ) from e
        except httpx.RequestError as e:
            logger.error("Request error fetching page '%s': %s", title, e)
            raise WikipediaAPIError(f"Request failed for '{title}'", cause=e) from e

        if response.status_code == 429:
            retry_after = self._parse_retry_after(response)
            logger.warning(
                "Rate limited fetching '%s', retry-after: %s", title, retry_after
            )
            raise RecoverableAPIError(
                f"Rate limited while fetching '{title}'",
                status_code=429,
                retry_after=retry_after,
            )

        if response.status_code >= 500:
            logger.warning("Server error %d fetching '%s'", response.status_code, title)
            raise RecoverableAPIError(
                f"Server error {response.status_code} fetching '{title}'",
                status_code=response.status_code,
            )

        if response.status_code >= 400:
            logger.error(
                "HTTP error %d fetching page '%s'", response.status_code, title
            )
            raise WikipediaAPIError(
                f"HTTP {response.status_code} fetching '{title}'",
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

    def _clean_text(self, text: str) -> str:
        """Clean extracted text by removing extra whitespace."""
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove common Wikipedia artifacts
        text = re.sub(r"\[edit\]", "", text)
        text = re.sub(r"\[\d+\]", "", text)
        return text.strip()

    def _normalize_title(self, title: str) -> str:
        """Normalize a Wikipedia title for comparison."""
        title = unquote(title)
        title = title.replace("_", " ")
        return title.strip().lower()

    async def traverse(self, title: str, depth: int) -> TraversalResult:
        """
        Traverse Wikipedia articles starting from the given title.

        Args:
            title: The starting Wikipedia article title.
            depth: The depth of traversal (0 = only starting article).

        Returns:
            TraversalResult containing all extracted texts and visited titles.
        """
        logger.info("Starting traversal from '%s' with depth %d", title, depth)
        result = TraversalResult()
        await self._traverse_recursive(title, depth, result)
        logger.info(
            "Traversal complete: %d articles fetched, %d visited, %d errors",
            len(result.texts),
            len(result.visited),
            len(result.errors),
        )
        return result

    async def _traverse_recursive(
        self,
        title: str,
        depth: int,
        result: TraversalResult,
    ) -> None:
        """Recursively traverse Wikipedia articles."""
        normalized_title = self._normalize_title(title)

        if normalized_title in result.visited:
            return

        result.visited.add(normalized_title)

        try:
            page = await self.fetch_page(title)
        except WikipediaFetchError as e:
            logger.warning("Failed to fetch '%s': %s", title, e)
            result.errors.append(TraversalError(title=title, error=str(e)))
            return

        if page is None:
            return

        result.texts.append(page.text)

        if depth <= 0:
            return

        # Traverse linked articles with rate limiting
        if page.links:
            await aiometer.run_on_each(
                partial(self._traverse_recursive, depth=depth - 1, result=result),
                page.links,
                max_at_once=self._settings.max_concurrent_requests,
                max_per_second=self._settings.max_requests_per_second,
            )


async def _main() -> None:
    """Test redirect by fetching 'UK' (redirects to 'United Kingdom')."""
    service = WikiRecursiveFetchService()
    try:
        page = await service.fetch_page("UK")
        if page:
            print(f"Requested: UK -> Resolved title: {page.title!r}")
            print(f"Text length: {len(page.text)} chars")
            print(f"First 200 chars: {page.text[:200]}...")
        else:
            print("Page not found")
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(_main())
