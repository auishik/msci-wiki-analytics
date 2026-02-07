from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from src.main import app
from src.models.schemas import WordFrequency, WordFrequencyResponse


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


class TestHealthEndpoint:
    def test_health_returns_healthy(self, client: TestClient) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestWordFrequencyEndpoint:
    @pytest.mark.asyncio
    async def test_word_frequency_returns_frequencies(
        self, async_client: AsyncClient
    ) -> None:
        mock_wiki_service = AsyncMock()
        mock_wiki_service.traverse.return_value = MagicMock(
            texts=["hello world hello python python python"]
        )

        mock_frequency_service = MagicMock()
        mock_frequency_service.calculate.return_value = WordFrequencyResponse(
            {
                "python": WordFrequency(count=3, percentage=50.0),
                "hello": WordFrequency(count=2, percentage=33.33),
                "world": WordFrequency(count=1, percentage=16.67),
            }
        )

        with (
            patch("src.main.wiki_service", mock_wiki_service),
            patch("src.main.frequency_service", mock_frequency_service),
        ):
            response = await async_client.get(
                "/word-frequency", params={"article": "Python", "depth": 0}
            )

        assert response.status_code == 200
        data = response.json()
        assert "python" in data
        assert data["python"]["count"] == 3

    @pytest.mark.asyncio
    async def test_word_frequency_article_not_found(
        self, async_client: AsyncClient
    ) -> None:
        mock_wiki_service = AsyncMock()
        mock_wiki_service.traverse.return_value = MagicMock(texts=[])

        mock_frequency_service = MagicMock()

        with (
            patch("src.main.wiki_service", mock_wiki_service),
            patch("src.main.frequency_service", mock_frequency_service),
        ):
            response = await async_client.get(
                "/word-frequency",
                params={"article": "NonexistentArticle123", "depth": 0},
            )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_word_frequency_validates_depth(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.get(
            "/word-frequency", params={"article": "Python", "depth": -1}
        )

        assert response.status_code == 422


class TestKeywordsEndpoint:
    @pytest.mark.asyncio
    async def test_keywords_returns_filtered_frequencies(
        self, async_client: AsyncClient
    ) -> None:
        mock_wiki_service = AsyncMock()
        mock_wiki_service.traverse.return_value = MagicMock(
            texts=["hello world hello python python python"]
        )

        mock_frequency_service = MagicMock()
        mock_frequency_service.calculate.return_value = WordFrequencyResponse(
            {
                "python": WordFrequency(count=3, percentage=75.0),
            }
        )

        with (
            patch("src.main.wiki_service", mock_wiki_service),
            patch("src.main.frequency_service", mock_frequency_service),
        ):
            response = await async_client.post(
                "/keywords",
                json={
                    "article": "Python",
                    "depth": 0,
                    "ignore_list": ["hello", "world"],
                    "percentile": 50,
                },
            )

        assert response.status_code == 200
        mock_frequency_service.calculate.assert_called_once_with(
            texts=["hello world hello python python python"],
            ignore_list=["hello", "world"],
            percentile=50,
        )

    @pytest.mark.asyncio
    async def test_keywords_article_not_found(self, async_client: AsyncClient) -> None:
        mock_wiki_service = AsyncMock()
        mock_wiki_service.traverse.return_value = MagicMock(texts=[])

        mock_frequency_service = MagicMock()

        with (
            patch("src.main.wiki_service", mock_wiki_service),
            patch("src.main.frequency_service", mock_frequency_service),
        ):
            response = await async_client.post(
                "/keywords",
                json={
                    "article": "NonexistentArticle123",
                    "depth": 0,
                    "ignore_list": [],
                    "percentile": 50,
                },
            )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_keywords_validates_request_body(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.post(
            "/keywords",
            json={
                "article": "Python",
                "depth": -1,
                "ignore_list": [],
                "percentile": 50,
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_keywords_validates_percentile_range(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.post(
            "/keywords",
            json={
                "article": "Python",
                "depth": 0,
                "ignore_list": [],
                "percentile": 100,
            },
        )

        assert response.status_code == 422


class TestExceptionHandlers:
    @pytest.mark.asyncio
    async def test_wikipedia_api_error_returns_502(
        self, async_client: AsyncClient
    ) -> None:
        from src.exceptions import WikipediaAPIError

        mock_wiki_service = AsyncMock()
        mock_wiki_service.traverse.side_effect = WikipediaAPIError("Connection failed")

        mock_frequency_service = MagicMock()

        with (
            patch("src.main.wiki_service", mock_wiki_service),
            patch("src.main.frequency_service", mock_frequency_service),
        ):
            response = await async_client.get(
                "/word-frequency", params={"article": "Python", "depth": 0}
            )

        assert response.status_code == 502
        assert "Connection failed" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_wikipedia_api_error_with_status_code(
        self, async_client: AsyncClient
    ) -> None:
        from src.exceptions import WikipediaAPIError

        mock_wiki_service = AsyncMock()
        mock_wiki_service.traverse.side_effect = WikipediaAPIError(
            "Rate limited", status_code=429
        )

        mock_frequency_service = MagicMock()

        with (
            patch("src.main.wiki_service", mock_wiki_service),
            patch("src.main.frequency_service", mock_frequency_service),
        ):
            response = await async_client.get(
                "/word-frequency", params={"article": "Python", "depth": 0}
            )

        assert response.status_code == 429

    @pytest.mark.asyncio
    async def test_wikipedia_parse_error_returns_502(
        self, async_client: AsyncClient
    ) -> None:
        from src.exceptions import WikipediaParseError

        mock_wiki_service = AsyncMock()
        mock_wiki_service.traverse.side_effect = WikipediaParseError("Invalid JSON")

        mock_frequency_service = MagicMock()

        with (
            patch("src.main.wiki_service", mock_wiki_service),
            patch("src.main.frequency_service", mock_frequency_service),
        ):
            response = await async_client.get(
                "/word-frequency", params={"article": "Python", "depth": 0}
            )

        assert response.status_code == 502
        assert "Invalid JSON" in response.json()["detail"]
