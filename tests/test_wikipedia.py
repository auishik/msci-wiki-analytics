from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

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
