import asyncio
import json
import logging
import re
import time
from urllib.parse import unquote

from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.exceptions import WikipediaAPIError, WikipediaFetchError, WikipediaParseError
from src.services.wiki_client import WikipediaAPIClient
from src.services.wiki_html_client import WikiHTMLClient

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter using semaphore for concurrency and token bucket for rate."""

    def __init__(self, max_concurrent: int, max_per_second: float) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_per_second = max_per_second
        self._min_interval = 1.0 / max_per_second
        self._last_request_time: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        await self._semaphore.acquire()
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_request_time = time.monotonic()

    def release(self) -> None:
        """Release the semaphore after request completes."""
        self._semaphore.release()


class PageContent(BaseModel):
    """Content extracted from a Wikipedia page."""

    title: str
    text: str
    links: list[str]
    redirects: list[str] = Field(default_factory=list)
    """Titles that redirect to this page (e.g., if 'UK' redirects to 'United Kingdom',
    redirects would contain 'UK')."""


class TraversalError(BaseModel):
    """Error encountered while traversing a Wikipedia article."""

    title: str
    error: str


class TraversalResult(BaseModel):
    """Result of traversing Wikipedia articles."""

    texts: list[str] = Field(default_factory=list)
    visited: set[str] = Field(default_factory=set)
    errors: list[TraversalError] = Field(default_factory=list)


class WikiRecursiveFetchService:
    """Service for fetching and recursively traversing Wikipedia articles."""

    def __init__(
        self,
        api_client: WikipediaAPIClient | None = None,
        html_client: WikiHTMLClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._api_client = api_client
        self._html_client = html_client
        self._owned_client: WikipediaAPIClient | WikiHTMLClient | None = None

        # Default to HTML client to avoid API rate limiting
        if api_client is None and html_client is None:
            self._html_client = WikiHTMLClient(settings=self._settings)
            self._owned_client = self._html_client

    async def close(self) -> None:
        """Close the client if we own it."""
        if self._owned_client is not None:
            await self._owned_client.close()

    async def fetch_page(self, title: str) -> PageContent | None:
        """
        Fetch a Wikipedia page's content and links.

        Dispatches to the HTML client (plain GET) or API client depending
        on which was configured.

        Args:
            title: The title of the Wikipedia article.

        Returns:
            PageContent with text and links, or None if page not found.

        Raises:
            WikipediaAPIError: If the request fails after retries.
            WikipediaParseError: If unable to parse the response.
        """
        if self._html_client is not None:
            return await self._fetch_page_html(title)
        if self._api_client is not None:
            return await self._fetch_page_api(title)
        raise RuntimeError("No client configured")

    # ------------------------------------------------------------------
    # HTML client path (plain GET to /wiki/<title>)
    # ------------------------------------------------------------------

    async def _fetch_page_html(self, title: str) -> PageContent | None:
        """Fetch a Wikipedia page by downloading its HTML directly."""
        assert self._html_client is not None
        logger.debug("Fetching page (HTML): %s", title)

        try:
            response = await self._html_client.get(title)
        except WikipediaAPIError as e:
            if e.status_code == 404:
                logger.warning("Page not found: %s", title)
                return None
            raise

        html = response.text
        soup = BeautifulSoup(html, "lxml")

        # Extract page title from the heading
        h1 = soup.find("h1", {"id": "firstHeading"})
        page_title = h1.get_text(strip=True) if isinstance(h1, Tag) else title

        # Scope to the parser-output content div inside mw-content-text
        # We must scope to #mw-content-text first because there may be multiple
        # mw-parser-output divs on the page (e.g., coordinates, empty divs)
        mw_content_text = soup.find("div", id="mw-content-text")
        if not isinstance(mw_content_text, Tag):
            logger.warning("No mw-content-text div found for page: %s", title)
            return None

        content_div = mw_content_text.find("div", class_="mw-parser-output")
        if not isinstance(content_div, Tag):
            logger.warning("No content div found for page: %s", title)
            return None

        # Extract links *before* decomposing elements
        links = self._extract_links_from_html(content_div)

        # Remove unwanted elements for text extraction
        for element in content_div.find_all(["script", "style", "table", "sup"]):
            element.decompose()
        for element in content_div.find_all(
            class_=[
                "navbox",
                "sidebar",
                "metadata",
                "mw-editsection",
                "reference",
                "toc",
                "catlinks",
            ]
        ):
            element.decompose()

        text = content_div.get_text(separator=" ", strip=True)
        text = self._clean_text(text)

        # Extract redirect source from page's JavaScript config
        redirects = self._extract_redirects_from_html(html)

        logger.debug(
            "Fetched page '%s' (HTML): %d chars, %d links, %d redirects",
            page_title,
            len(text),
            len(links),
            len(redirects),
        )
        return PageContent(
            title=page_title,
            text=text,
            links=links,
            redirects=redirects,
        )

    def _extract_links_from_html(self, content_div: Tag) -> list[str]:
        """Extract internal Wikipedia article links from HTML content."""
        links: list[str] = []
        seen: set[str] = set()

        for a_tag in content_div.find_all("a", href=True):
            href = a_tag["href"]
            if not isinstance(href, str) or not href.startswith("/wiki/"):
                continue

            # Strip /wiki/ prefix
            article_path = href[6:]

            # Remove fragment
            if "#" in article_path:
                article_path = article_path.split("#")[0]

            if not article_path:
                continue

            # Skip namespace pages (File:, Wikipedia:, Help:, etc.)
            if ":" in unquote(article_path):
                continue

            # Decode and normalize
            title = unquote(article_path).replace("_", " ")
            normalized = title.lower()

            if normalized not in seen:
                seen.add(normalized)
                links.append(title)

        return links

    def _extract_redirects_from_html(self, html: str) -> list[str]:
        """Extract redirect source from HTML page config (wgRedirectedFrom).

        Wikipedia handles article redirects internally (not via HTTP 301/302),
        embedding the original title in the page's JavaScript config.
        """
        redirects: list[str] = []

        # Look for wgRedirectedFrom in the JavaScript config
        # Format: "wgRedirectedFrom":"UK"
        match = re.search(r'"wgRedirectedFrom"\s*:\s*"([^"]+)"', html)
        if match:
            redirect_from = match.group(1)
            # Unescape any JSON string escapes
            try:
                redirect_from = json.loads(f'"{redirect_from}"')
            except json.JSONDecodeError:
                pass
            redirects.append(redirect_from)

        return redirects

    # ------------------------------------------------------------------
    # API client path (MediaWiki API /w/api.php)
    # ------------------------------------------------------------------

    async def _fetch_page_api(self, title: str) -> PageContent | None:
        """Fetch a Wikipedia page via the MediaWiki API."""
        assert self._api_client is not None
        logger.debug("Fetching page (API): %s", title)

        params = {
            "action": "parse",
            "page": title,
            "format": "json",
            "prop": "text|links",
            "redirects": "1",
        }

        response = await self._api_client.get(params)

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

        # Extract redirect sources (titles that redirect to this page)
        redirects: list[str] = []
        raw_redirects = parse_data.get("redirects", [])
        for redirect in raw_redirects:
            if "from" in redirect:
                redirects.append(redirect["from"])

        logger.debug(
            "Fetched page '%s' (API): %d chars, %d links, %d redirects",
            title,
            len(text),
            len(links),
            len(redirects),
        )
        return PageContent(
            title=parse_data.get("title", title),
            text=text,
            links=links,
            redirects=redirects,
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
        rate_limiter = RateLimiter(
            max_concurrent=self._settings.max_concurrent_requests,
            max_per_second=self._settings.max_requests_per_second,
        )
        await self._traverse_recursive(title, depth, result, rate_limiter)
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
        rate_limiter: RateLimiter,
    ) -> None:
        """Recursively traverse Wikipedia articles."""
        normalized_title = self._normalize_title(title)

        if normalized_title in result.visited:
            return

        result.visited.add(normalized_title)

        await rate_limiter.acquire()
        try:
            page = await self.fetch_page(title)
        except WikipediaFetchError as e:
            logger.warning("Failed to fetch '%s': %s", title, e)
            result.errors.append(TraversalError(title=title, error=str(e)))
            return
        finally:
            rate_limiter.release()

        if page is None:
            return

        # Add the resolved page title and any redirects to visited set
        # This prevents re-fetching the same page via different aliases
        result.visited.add(self._normalize_title(page.title))
        for redirect in page.redirects:
            result.visited.add(self._normalize_title(redirect))

        result.texts.append(page.text)

        if depth <= 0:
            return

        # Traverse linked articles concurrently with shared rate limiter
        if page.links:
            tasks = [
                self._traverse_recursive(link, depth - 1, result, rate_limiter)
                for link in page.links
            ]
            await asyncio.gather(*tasks)


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
