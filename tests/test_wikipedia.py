from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.config import Settings
from src.services.wikipedia import WikiRecursiveFetchService


def make_wiki_response(
    title: str,
    html: str,
    links: list[dict[str, str | int | None]],
) -> dict:
    """Create a mock Wikipedia API response."""
    return {
        "parse": {
            "title": title,
            "text": {"*": html},
            "links": links,
        }
    }


def make_link(title: str, exists: bool = True) -> dict:
    """Create a mock Wikipedia link entry."""
    return {"*": title, "ns": 0, "exists": "" if exists else None}


def make_mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock HTTP response with proper attributes."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = json_data
    return mock_response


class TestWikiRecursiveFetchService:
    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        return AsyncMock(spec=httpx.AsyncClient)

    @pytest.fixture
    def service(self, mock_client: AsyncMock) -> WikiRecursiveFetchService:
        service = WikiRecursiveFetchService(client=mock_client)
        return service

    @pytest.mark.asyncio
    async def test_fetch_page_returns_content(
        self, service: WikiRecursiveFetchService, mock_client: AsyncMock
    ) -> None:
        mock_response = make_mock_response(
            make_wiki_response(
                title="Python",
                html="<p>Python is a programming language.</p>",
                links=[make_link("Java"), make_link("Ruby")],
            )
        )
        mock_client.get.return_value = mock_response

        result = await service.fetch_page("Python")

        assert result is not None
        assert result.title == "Python"
        assert "Python is a programming language" in result.text
        assert result.links == ["Java", "Ruby"]

    @pytest.mark.asyncio
    async def test_fetch_page_returns_none_on_error(
        self, service: WikiRecursiveFetchService, mock_client: AsyncMock
    ) -> None:
        mock_response = make_mock_response({"error": {"code": "missingtitle"}})
        mock_client.get.return_value = mock_response

        result = await service.fetch_page("NonexistentPage12345")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_page_filters_non_article_links(
        self, service: WikiRecursiveFetchService, mock_client: AsyncMock
    ) -> None:
        mock_response = make_mock_response(
            make_wiki_response(
                title="Test",
                html="<p>Test content</p>",
                links=[
                    {"*": "Article", "ns": 0, "exists": ""},
                    {"*": "Category:Test", "ns": 14, "exists": ""},
                    {"*": "Nonexistent", "ns": 0, "exists": None},
                ],
            )
        )
        mock_client.get.return_value = mock_response

        result = await service.fetch_page("Test")

        assert result is not None
        assert result.links == ["Article"]

    @pytest.mark.asyncio
    async def test_traverse_depth_zero_returns_single_article(
        self, service: WikiRecursiveFetchService, mock_client: AsyncMock
    ) -> None:
        mock_response = make_mock_response(
            make_wiki_response(
                title="Python",
                html="<p>Python content</p>",
                links=[make_link("Java")],
            )
        )
        mock_client.get.return_value = mock_response

        result = await service.traverse("Python", depth=0)

        assert len(result.texts) == 1
        assert len(result.visited) == 1
        assert "python" in result.visited
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_traverse_depth_one_fetches_linked_articles(
        self, service: WikiRecursiveFetchService, mock_client: AsyncMock
    ) -> None:
        responses = {
            "Python": make_wiki_response(
                title="Python",
                html="<p>Python content</p>",
                links=[make_link("Java"), make_link("Ruby")],
            ),
            "Java": make_wiki_response(
                title="Java",
                html="<p>Java content</p>",
                links=[],
            ),
            "Ruby": make_wiki_response(
                title="Ruby",
                html="<p>Ruby content</p>",
                links=[],
            ),
        }

        async def get_response(url: str, params: dict) -> MagicMock:
            title = params["page"]
            json_data = responses.get(title, {"error": {"code": "missingtitle"}})
            return make_mock_response(json_data)

        mock_client.get.side_effect = get_response

        result = await service.traverse("Python", depth=1)

        assert len(result.texts) == 3
        assert len(result.visited) == 3
        assert "python" in result.visited
        assert "java" in result.visited
        assert "ruby" in result.visited

    @pytest.mark.asyncio
    async def test_traverse_avoids_revisiting_articles(
        self, service: WikiRecursiveFetchService, mock_client: AsyncMock
    ) -> None:
        responses = {
            "A": make_wiki_response(
                title="A",
                html="<p>A content</p>",
                links=[make_link("B")],
            ),
            "B": make_wiki_response(
                title="B",
                html="<p>B content</p>",
                links=[make_link("A")],
            ),
        }

        async def get_response(url: str, params: dict) -> MagicMock:
            title = params["page"]
            json_data = responses.get(title, {"error": {"code": "missingtitle"}})
            return make_mock_response(json_data)

        mock_client.get.side_effect = get_response

        result = await service.traverse("A", depth=2)

        assert len(result.texts) == 2
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_traverse_handles_missing_pages(
        self, service: WikiRecursiveFetchService, mock_client: AsyncMock
    ) -> None:
        responses = {
            "A": make_wiki_response(
                title="A",
                html="<p>A content</p>",
                links=[make_link("Missing"), make_link("B")],
            ),
            "B": make_wiki_response(
                title="B",
                html="<p>B content</p>",
                links=[],
            ),
        }

        async def get_response(url: str, params: dict) -> MagicMock:
            title = params["page"]
            json_data = responses.get(title, {"error": {"code": "missingtitle"}})
            return make_mock_response(json_data)

        mock_client.get.side_effect = get_response

        result = await service.traverse("A", depth=1)

        assert len(result.texts) == 2


class TestCleanText:
    def test_removes_extra_whitespace(self) -> None:
        service = WikiRecursiveFetchService()
        result = service._clean_text("hello    world\n\ntest")
        assert result == "hello world test"

    def test_removes_edit_markers(self) -> None:
        service = WikiRecursiveFetchService()
        result = service._clean_text("Section[edit] content")
        assert result == "Section content"

    def test_removes_citation_numbers(self) -> None:
        service = WikiRecursiveFetchService()
        result = service._clean_text("Fact[1] and another[23]")
        assert result == "Fact and another"


class TestNormalizeTitle:
    def test_normalizes_underscores(self) -> None:
        service = WikiRecursiveFetchService()
        result = service._normalize_title("Python_(programming_language)")
        assert result == "python (programming language)"

    def test_handles_url_encoding(self) -> None:
        service = WikiRecursiveFetchService()
        result = service._normalize_title("C%2B%2B")
        assert result == "c++"

    def test_lowercases_title(self) -> None:
        service = WikiRecursiveFetchService()
        result = service._normalize_title("PYTHON")
        assert result == "python"


class TestRetryBehavior:
    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        return AsyncMock(spec=httpx.AsyncClient)

    @pytest.fixture
    def settings(self) -> Settings:
        return Settings(
            retry_max_attempts=3,
            retry_min_wait=0.01,
            retry_max_wait=0.1,
            retry_multiplier=1.0,
            retry_jitter_max=0.01,
        )

    @pytest.fixture
    def service(
        self, mock_client: AsyncMock, settings: Settings
    ) -> WikiRecursiveFetchService:
        return WikiRecursiveFetchService(client=mock_client, settings=settings)

    @pytest.mark.asyncio
    async def test_retries_on_timeout(
        self, service: WikiRecursiveFetchService, mock_client: AsyncMock
    ) -> None:
        mock_client.get.side_effect = [
            httpx.TimeoutException("timeout"),
            httpx.TimeoutException("timeout"),
            make_mock_response(make_wiki_response("Test", "<p>Content</p>", [])),
        ]

        result = await service.fetch_page("Test")

        assert result is not None
        assert result.title == "Test"
        assert mock_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(
        self, service: WikiRecursiveFetchService, mock_client: AsyncMock
    ) -> None:
        mock_client.get.side_effect = [
            httpx.ConnectError("connection failed"),
            make_mock_response(make_wiki_response("Test", "<p>Content</p>", [])),
        ]

        result = await service.fetch_page("Test")

        assert result is not None
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_429_rate_limit(
        self, service: WikiRecursiveFetchService, mock_client: AsyncMock
    ) -> None:
        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {}

        mock_client.get.side_effect = [
            rate_limit_response,
            make_mock_response(make_wiki_response("Test", "<p>Content</p>", [])),
        ]

        result = await service.fetch_page("Test")

        assert result is not None
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_500_server_error(
        self, service: WikiRecursiveFetchService, mock_client: AsyncMock
    ) -> None:
        server_error_response = MagicMock()
        server_error_response.status_code = 500

        mock_client.get.side_effect = [
            server_error_response,
            make_mock_response(make_wiki_response("Test", "<p>Content</p>", [])),
        ]

        result = await service.fetch_page("Test")

        assert result is not None
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_uses_retry_after_header(
        self, service: WikiRecursiveFetchService, mock_client: AsyncMock
    ) -> None:
        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {"Retry-After": "0.05"}

        mock_client.get.side_effect = [
            rate_limit_response,
            make_mock_response(make_wiki_response("Test", "<p>Content</p>", [])),
        ]

        result = await service.fetch_page("Test")

        assert result is not None
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_max_retries_exceeded(
        self, service: WikiRecursiveFetchService, mock_client: AsyncMock
    ) -> None:
        from src.exceptions import RecoverableAPIError

        mock_client.get.side_effect = httpx.TimeoutException("timeout")

        with pytest.raises(RecoverableAPIError, match="Timeout"):
            await service.fetch_page("Test")

        assert mock_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_does_not_retry_on_404(
        self, service: WikiRecursiveFetchService, mock_client: AsyncMock
    ) -> None:
        from src.exceptions import WikipediaAPIError

        not_found_response = MagicMock()
        not_found_response.status_code = 404

        mock_client.get.return_value = not_found_response

        with pytest.raises(WikipediaAPIError, match="404"):
            await service.fetch_page("Test")

        assert mock_client.get.call_count == 1


class TestErrorCollection:
    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        return AsyncMock(spec=httpx.AsyncClient)

    @pytest.fixture
    def settings(self) -> Settings:
        return Settings(
            retry_max_attempts=1,
            retry_min_wait=0.01,
            retry_max_wait=0.01,
        )

    @pytest.fixture
    def service(
        self, mock_client: AsyncMock, settings: Settings
    ) -> WikiRecursiveFetchService:
        return WikiRecursiveFetchService(client=mock_client, settings=settings)

    @pytest.mark.asyncio
    async def test_traverse_collects_errors_and_continues(
        self, service: WikiRecursiveFetchService, mock_client: AsyncMock
    ) -> None:
        responses = {
            "A": make_wiki_response(
                title="A",
                html="<p>A content</p>",
                links=[make_link("B"), make_link("C")],
            ),
            "C": make_wiki_response(
                title="C",
                html="<p>C content</p>",
                links=[],
            ),
        }

        async def get_response(url: str, params: dict) -> MagicMock:
            title = params["page"]
            if title == "B":
                raise httpx.TimeoutException("timeout")
            json_data = responses.get(title, {"error": {"code": "missingtitle"}})
            return make_mock_response(json_data)

        mock_client.get.side_effect = get_response

        result = await service.traverse("A", depth=1)

        assert len(result.texts) == 2  # A and C succeeded
        assert len(result.errors) == 1  # B failed
        assert result.errors[0].title == "B"
        assert "Timeout" in result.errors[0].error

    @pytest.mark.asyncio
    async def test_traverse_returns_partial_results_on_errors(
        self, service: WikiRecursiveFetchService, mock_client: AsyncMock
    ) -> None:
        responses = {
            "A": make_wiki_response(
                title="A",
                html="<p>A content</p>",
                links=[make_link("B"), make_link("C"), make_link("D")],
            ),
            "B": make_wiki_response(
                title="B",
                html="<p>B content</p>",
                links=[],
            ),
        }

        async def get_response(url: str, params: dict) -> MagicMock:
            title = params["page"]
            if title in ("C", "D"):
                raise httpx.ConnectError("connection failed")
            json_data = responses.get(title, {"error": {"code": "missingtitle"}})
            return make_mock_response(json_data)

        mock_client.get.side_effect = get_response

        result = await service.traverse("A", depth=1)

        assert len(result.texts) == 2  # A and B succeeded
        assert len(result.errors) == 2  # C and D failed
        error_titles = {e.title for e in result.errors}
        assert error_titles == {"C", "D"}

    @pytest.mark.asyncio
    async def test_traverse_empty_errors_on_success(
        self, service: WikiRecursiveFetchService, mock_client: AsyncMock
    ) -> None:
        mock_response = make_mock_response(
            make_wiki_response(
                title="A",
                html="<p>A content</p>",
                links=[],
            )
        )
        mock_client.get.return_value = mock_response

        result = await service.traverse("A", depth=0)

        assert len(result.texts) == 1
        assert len(result.errors) == 0
