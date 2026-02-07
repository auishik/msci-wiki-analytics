import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from urllib.parse import unquote

import httpx
from bs4 import BeautifulSoup

from src.exceptions import WikipediaAPIError, WikipediaParseError

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Content extracted from a Wikipedia page."""

    title: str
    text: str
    links: list[str]


@dataclass
class TraversalResult:
    """Result of traversing Wikipedia articles."""

    texts: list[str] = field(default_factory=list)
    visited: set[str] = field(default_factory=set)


class WikiRecursiveFetchService:
    """Service for fetching and recursively traversing Wikipedia articles."""

    BASE_URL = "https://en.wikipedia.org/w/api.php"
    DEFAULT_USER_AGENT = (
        "msci-wiki-analytics/0.1.0 "
        "(https://github.com/user/msci-wiki-analytics; contact@example.com)"
    )

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        user_agent: str | None = None,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self._user_agent = user_agent or self.DEFAULT_USER_AGENT

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "User-Agent": self._user_agent,
                    "Api-User-Agent": self._user_agent,
                },
                timeout=30.0,
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
            WikipediaAPIError: If the API request fails.
            WikipediaParseError: If unable to parse the response.
        """
        logger.debug("Fetching page: %s", title)
        client = await self._get_client()

        params = {
            "action": "parse",
            "page": title,
            "format": "json",
            "prop": "text|links",
            "redirects": "1",
        }

        try:
            response = await client.get(self.BASE_URL, params=params)
        except httpx.TimeoutException as e:
            logger.error("Timeout fetching page '%s': %s", title, e)
            raise WikipediaAPIError(f"Timeout fetching page '{title}'", cause=e) from e
        except httpx.RequestError as e:
            logger.error("Request error fetching page '%s': %s", title, e)
            raise WikipediaAPIError(f"Request failed for '{title}'", cause=e) from e

        if response.status_code == 429:
            logger.warning("Rate limited while fetching page '%s'", title)
            raise WikipediaAPIError(
                f"Rate limited while fetching '{title}'", status_code=429
            )

        if response.status_code >= 400:
            logger.error(
                "HTTP error %d fetching page '%s'", response.status_code, title
            )
            raise WikipediaAPIError(
                f"HTTP {response.status_code} fetching '{title}'",
                status_code=response.status_code,
            )

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
            "Traversal complete: %d articles fetched, %d total visited",
            len(result.texts),
            len(result.visited),
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

        page = await self.fetch_page(title)
        if page is None:
            return

        result.texts.append(page.text)

        if depth <= 0:
            return

        # Traverse linked articles concurrently
        # Each task handles its own visited check
        tasks = [
            self._traverse_recursive(link, depth - 1, result) for link in page.links
        ]
        if tasks:
            await asyncio.gather(*tasks)
