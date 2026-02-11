from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import Settings
from src.exceptions import RecoverableAPIError, WikipediaAPIError
from src.services.wiki_client import WikipediaAPIClient
from src.services.wiki_html_client import WikiHTMLClient
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
    def mock_api_client(self) -> AsyncMock:
        return AsyncMock(spec=WikipediaAPIClient)

    @pytest.fixture
    def service(self, mock_api_client: AsyncMock) -> WikiRecursiveFetchService:
        service = WikiRecursiveFetchService(api_client=mock_api_client)
        return service

    @pytest.mark.asyncio
    async def test_fetch_page_returns_content(
        self, service: WikiRecursiveFetchService, mock_api_client: AsyncMock
    ) -> None:
        mock_response = make_mock_response(
            make_wiki_response(
                title="Python",
                html="<p>Python is a programming language.</p>",
                links=[make_link("Java"), make_link("Ruby")],
            )
        )
        mock_api_client.get.return_value = mock_response

        result = await service.fetch_page("Python")

        assert result is not None
        assert result.title == "Python"
        assert "Python is a programming language" in result.text
        assert result.links == ["Java", "Ruby"]

    @pytest.mark.asyncio
    async def test_fetch_page_returns_none_on_error(
        self, service: WikiRecursiveFetchService, mock_api_client: AsyncMock
    ) -> None:
        mock_response = make_mock_response({"error": {"code": "missingtitle"}})
        mock_api_client.get.return_value = mock_response

        result = await service.fetch_page("NonexistentPage12345")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_page_filters_non_article_links(
        self, service: WikiRecursiveFetchService, mock_api_client: AsyncMock
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
        mock_api_client.get.return_value = mock_response

        result = await service.fetch_page("Test")

        assert result is not None
        assert result.links == ["Article"]

    @pytest.mark.asyncio
    async def test_traverse_depth_zero_returns_single_article(
        self, service: WikiRecursiveFetchService, mock_api_client: AsyncMock
    ) -> None:
        mock_response = make_mock_response(
            make_wiki_response(
                title="Python",
                html="<p>Python content</p>",
                links=[make_link("Java")],
            )
        )
        mock_api_client.get.return_value = mock_response

        result = await service.traverse("Python", depth=0)

        assert len(result.texts) == 1
        assert len(result.visited) == 1
        assert "python" in result.visited
        mock_api_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_traverse_depth_one_fetches_linked_articles(
        self, service: WikiRecursiveFetchService, mock_api_client: AsyncMock
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

        async def get_response(params: dict) -> MagicMock:
            title = params["page"]
            json_data = responses.get(title, {"error": {"code": "missingtitle"}})
            return make_mock_response(json_data)

        mock_api_client.get.side_effect = get_response

        result = await service.traverse("Python", depth=1)

        assert len(result.texts) == 3
        assert len(result.visited) == 3
        assert "python" in result.visited
        assert "java" in result.visited
        assert "ruby" in result.visited

    @pytest.mark.asyncio
    async def test_traverse_avoids_revisiting_articles(
        self, service: WikiRecursiveFetchService, mock_api_client: AsyncMock
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

        async def get_response(params: dict) -> MagicMock:
            title = params["page"]
            json_data = responses.get(title, {"error": {"code": "missingtitle"}})
            return make_mock_response(json_data)

        mock_api_client.get.side_effect = get_response

        result = await service.traverse("A", depth=2)

        assert len(result.texts) == 2
        assert mock_api_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_traverse_handles_missing_pages(
        self, service: WikiRecursiveFetchService, mock_api_client: AsyncMock
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

        async def get_response(params: dict) -> MagicMock:
            title = params["page"]
            json_data = responses.get(title, {"error": {"code": "missingtitle"}})
            return make_mock_response(json_data)

        mock_api_client.get.side_effect = get_response

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

    def test_normalizes_unicode_forms(self) -> None:
        """Precomposed and decomposed Unicode should normalize to the same value."""
        service = WikiRecursiveFetchService()
        # Precomposed: ñ is U+00F1 (single character)
        precomposed = "Ñāṇamoli Bhikkhu"
        # Decomposed: ñ is U+004E + U+0303 (N + combining tilde)
        decomposed = "N\u0303a\u0304n\u0323amoli Bhikkhu"

        result_precomposed = service._normalize_title(precomposed)
        result_decomposed = service._normalize_title(decomposed)

        assert result_precomposed == result_decomposed


class TestErrorCollection:
    @pytest.fixture
    def mock_api_client(self) -> AsyncMock:
        return AsyncMock(spec=WikipediaAPIClient)

    @pytest.fixture
    def settings(self) -> Settings:
        return Settings(
            retry_max_attempts=1,
            retry_min_wait=0.01,
            retry_max_wait=0.01,
        )

    @pytest.fixture
    def service(
        self, mock_api_client: AsyncMock, settings: Settings
    ) -> WikiRecursiveFetchService:
        return WikiRecursiveFetchService(api_client=mock_api_client, settings=settings)

    @pytest.mark.asyncio
    async def test_traverse_collects_errors_and_continues(
        self, service: WikiRecursiveFetchService, mock_api_client: AsyncMock
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

        async def get_response(params: dict) -> MagicMock:
            title = params["page"]
            if title == "B":
                raise RecoverableAPIError("Timeout for 'B'")
            json_data = responses.get(title, {"error": {"code": "missingtitle"}})
            return make_mock_response(json_data)

        mock_api_client.get.side_effect = get_response

        result = await service.traverse("A", depth=1)

        assert len(result.texts) == 2  # A and C succeeded
        assert len(result.errors) == 1  # B failed
        assert result.errors[0].title == "B"
        assert "Timeout" in result.errors[0].error

    @pytest.mark.asyncio
    async def test_traverse_returns_partial_results_on_errors(
        self, service: WikiRecursiveFetchService, mock_api_client: AsyncMock
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

        async def get_response(params: dict) -> MagicMock:
            title = params["page"]
            if title in ("C", "D"):
                raise WikipediaAPIError(f"Connection failed for '{title}'")
            json_data = responses.get(title, {"error": {"code": "missingtitle"}})
            return make_mock_response(json_data)

        mock_api_client.get.side_effect = get_response

        result = await service.traverse("A", depth=1)

        assert len(result.texts) == 2  # A and B succeeded
        assert len(result.errors) == 2  # C and D failed
        error_titles = {e.title for e in result.errors}
        assert error_titles == {"C", "D"}

    @pytest.mark.asyncio
    async def test_traverse_empty_errors_on_success(
        self, service: WikiRecursiveFetchService, mock_api_client: AsyncMock
    ) -> None:
        mock_response = make_mock_response(
            make_wiki_response(
                title="A",
                html="<p>A content</p>",
                links=[],
            )
        )
        mock_api_client.get.return_value = mock_response

        result = await service.traverse("A", depth=0)

        assert len(result.texts) == 1
        assert len(result.errors) == 0


# ------------------------------------------------------------------
# Tests for the HTML client path
# ------------------------------------------------------------------


def make_wiki_html_page(
    title: str,
    body_html: str,
    link_titles: list[str] | None = None,
) -> str:
    """Build a minimal Wikipedia-like HTML page for testing."""
    links_html = ""
    if link_titles:
        links_html = " ".join(
            f'<a href="/wiki/{t.replace(" ", "_")}">{t}</a>' for t in link_titles
        )
    return (
        f"<html><head><title>{title}</title></head>"
        f'<body><h1 id="firstHeading">{title}</h1>'
        f'<div id="mw-content-text"><div class="mw-parser-output">'
        f"{body_html} {links_html}"
        f"</div></div></body></html>"
    )


def make_html_mock_response(html: str, status_code: int = 200) -> MagicMock:
    """Create a mock HTTP response carrying HTML text."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.text = html
    return mock


class TestHTMLFetchPath:
    @pytest.fixture
    def mock_html_client(self) -> AsyncMock:
        return AsyncMock(spec=WikiHTMLClient)

    @pytest.fixture
    def service(self, mock_html_client: AsyncMock) -> WikiRecursiveFetchService:
        return WikiRecursiveFetchService(html_client=mock_html_client)

    @pytest.mark.asyncio
    async def test_fetch_page_html_returns_content(
        self, service: WikiRecursiveFetchService, mock_html_client: AsyncMock
    ) -> None:
        page_html = make_wiki_html_page(
            title="Python",
            body_html="<p>Python is a programming language.</p>",
            link_titles=["Java", "Ruby"],
        )
        mock_html_client.get.return_value = make_html_mock_response(page_html)

        result = await service.fetch_page("Python")

        assert result is not None
        assert result.title == "Python"
        assert "Python is a programming language" in result.text
        assert "Java" in result.links
        assert "Ruby" in result.links

    @pytest.mark.asyncio
    async def test_fetch_page_html_returns_none_on_404(
        self, service: WikiRecursiveFetchService, mock_html_client: AsyncMock
    ) -> None:
        mock_html_client.get.side_effect = WikipediaAPIError(
            "HTTP 404 for 'Nonexistent'", status_code=404
        )

        result = await service.fetch_page("Nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_page_html_skips_namespace_links(
        self, service: WikiRecursiveFetchService, mock_html_client: AsyncMock
    ) -> None:
        page_html = (
            '<html><body><h1 id="firstHeading">Test</h1>'
            '<div id="mw-content-text"><div class="mw-parser-output">'
            "<p>Content</p>"
            '<a href="/wiki/Article">Article</a>'
            '<a href="/wiki/File:Image.png">Image</a>'
            '<a href="/wiki/Wikipedia:About">About</a>'
            '<a href="/wiki/Help:Contents">Help</a>'
            "</div></div></body></html>"
        )
        mock_html_client.get.return_value = make_html_mock_response(page_html)

        result = await service.fetch_page("Test")

        assert result is not None
        assert result.links == ["Article"]

    @pytest.mark.asyncio
    async def test_fetch_page_html_deduplicates_links(
        self, service: WikiRecursiveFetchService, mock_html_client: AsyncMock
    ) -> None:
        page_html = (
            '<html><body><h1 id="firstHeading">Test</h1>'
            '<div id="mw-content-text"><div class="mw-parser-output">'
            "<p>Content</p>"
            '<a href="/wiki/Python">Python</a>'
            '<a href="/wiki/Python#History">Python history</a>'
            '<a href="/wiki/python">python</a>'
            "</div></div></body></html>"
        )
        mock_html_client.get.return_value = make_html_mock_response(page_html)

        result = await service.fetch_page("Test")

        assert result is not None
        # All three links refer to the same article
        assert len(result.links) == 1
        assert result.links[0] == "Python"

    @pytest.mark.asyncio
    async def test_fetch_page_html_strips_fragments(
        self, service: WikiRecursiveFetchService, mock_html_client: AsyncMock
    ) -> None:
        page_html = (
            '<html><body><h1 id="firstHeading">Test</h1>'
            '<div id="mw-content-text"><div class="mw-parser-output">'
            "<p>Content</p>"
            '<a href="/wiki/Java#Syntax">Java</a>'
            "</div></div></body></html>"
        )
        mock_html_client.get.return_value = make_html_mock_response(page_html)

        result = await service.fetch_page("Test")

        assert result is not None
        assert result.links == ["Java"]

    @pytest.mark.asyncio
    async def test_traverse_with_html_client(
        self, service: WikiRecursiveFetchService, mock_html_client: AsyncMock
    ) -> None:
        pages = {
            "Python": make_wiki_html_page(
                title="Python",
                body_html="<p>Python content</p>",
                link_titles=["Java"],
            ),
            "Java": make_wiki_html_page(
                title="Java",
                body_html="<p>Java content</p>",
            ),
        }

        async def get_html(title: str) -> MagicMock:
            if title in pages:
                return make_html_mock_response(pages[title])
            raise WikipediaAPIError(f"HTTP 404 for '{title}'", status_code=404)

        mock_html_client.get.side_effect = get_html

        result = await service.traverse("Python", depth=1)

        assert len(result.texts) == 2
        assert "python" in result.visited
        assert "java" in result.visited

    @pytest.mark.asyncio
    async def test_fetch_page_html_removes_tables(
        self, service: WikiRecursiveFetchService, mock_html_client: AsyncMock
    ) -> None:
        page_html = make_wiki_html_page(
            title="Test",
            body_html=(
                "<p>Visible content.</p>" "<table><tr><td>Table data</td></tr></table>"
            ),
        )
        mock_html_client.get.return_value = make_html_mock_response(page_html)

        result = await service.fetch_page("Test")

        assert result is not None
        assert "Visible content" in result.text
        assert "Table data" not in result.text

    @pytest.mark.asyncio
    async def test_fetch_page_html_reraises_non_404_errors(
        self, service: WikiRecursiveFetchService, mock_html_client: AsyncMock
    ) -> None:
        mock_html_client.get.side_effect = RecoverableAPIError(
            "Timeout for 'Test'", status_code=None
        )

        with pytest.raises(RecoverableAPIError, match="Timeout"):
            await service.fetch_page("Test")
