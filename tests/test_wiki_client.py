from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.config import Settings
from src.exceptions import RecoverableAPIError, WikipediaAPIError
from src.services.wiki_client import WikipediaAPIClient


def make_mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock HTTP response with proper attributes."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = json_data
    return mock_response


class TestWikipediaAPIClient:
    @pytest.fixture
    def mock_http_client(self) -> AsyncMock:
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
    def client(
        self, mock_http_client: AsyncMock, settings: Settings
    ) -> WikipediaAPIClient:
        return WikipediaAPIClient(client=mock_http_client, settings=settings)

    @pytest.mark.asyncio
    async def test_get_returns_response(
        self, client: WikipediaAPIClient, mock_http_client: AsyncMock
    ) -> None:
        mock_response = make_mock_response({"parse": {"title": "Test"}})
        mock_http_client.get.return_value = mock_response

        result = await client.get({"action": "parse", "page": "Test", "format": "json"})

        assert result.json() == {"parse": {"title": "Test"}}
        mock_http_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_retries_on_timeout(
        self, client: WikipediaAPIClient, mock_http_client: AsyncMock
    ) -> None:
        mock_http_client.get.side_effect = [
            httpx.TimeoutException("timeout"),
            httpx.TimeoutException("timeout"),
            make_mock_response({"parse": {"title": "Test"}}),
        ]

        result = await client.get({"action": "parse", "page": "Test", "format": "json"})

        assert result.json() == {"parse": {"title": "Test"}}
        assert mock_http_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(
        self, client: WikipediaAPIClient, mock_http_client: AsyncMock
    ) -> None:
        mock_http_client.get.side_effect = [
            httpx.ConnectError("connection failed"),
            make_mock_response({"parse": {"title": "Test"}}),
        ]

        result = await client.get({"action": "parse", "page": "Test", "format": "json"})

        assert result is not None
        assert mock_http_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_429_rate_limit(
        self, client: WikipediaAPIClient, mock_http_client: AsyncMock
    ) -> None:
        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {}

        mock_http_client.get.side_effect = [
            rate_limit_response,
            make_mock_response({"parse": {"title": "Test"}}),
        ]

        result = await client.get({"action": "parse", "page": "Test", "format": "json"})

        assert result is not None
        assert mock_http_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_500_server_error(
        self, client: WikipediaAPIClient, mock_http_client: AsyncMock
    ) -> None:
        server_error_response = MagicMock()
        server_error_response.status_code = 500

        mock_http_client.get.side_effect = [
            server_error_response,
            make_mock_response({"parse": {"title": "Test"}}),
        ]

        result = await client.get({"action": "parse", "page": "Test", "format": "json"})

        assert result is not None
        assert mock_http_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_uses_retry_after_header(
        self, client: WikipediaAPIClient, mock_http_client: AsyncMock
    ) -> None:
        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {"Retry-After": "0.05"}

        mock_http_client.get.side_effect = [
            rate_limit_response,
            make_mock_response({"parse": {"title": "Test"}}),
        ]

        result = await client.get({"action": "parse", "page": "Test", "format": "json"})

        assert result is not None
        assert mock_http_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_max_retries_exceeded(
        self, client: WikipediaAPIClient, mock_http_client: AsyncMock
    ) -> None:
        mock_http_client.get.side_effect = httpx.TimeoutException("timeout")

        with pytest.raises(RecoverableAPIError, match="Timeout"):
            await client.get({"action": "parse", "page": "Test", "format": "json"})

        assert mock_http_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_does_not_retry_on_404(
        self, client: WikipediaAPIClient, mock_http_client: AsyncMock
    ) -> None:
        not_found_response = MagicMock()
        not_found_response.status_code = 404

        mock_http_client.get.return_value = not_found_response

        with pytest.raises(WikipediaAPIError, match="404"):
            await client.get({"action": "parse", "page": "Test", "format": "json"})

        assert mock_http_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_close_closes_owned_client(self) -> None:
        """Client created internally should be closed."""
        client = WikipediaAPIClient()
        # Force client creation
        await client._get_client()
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_does_not_close_injected_client(
        self, mock_http_client: AsyncMock
    ) -> None:
        """Injected client should not be closed by WikipediaAPIClient."""
        client = WikipediaAPIClient(client=mock_http_client)
        await client.close()
        mock_http_client.aclose.assert_not_called()
