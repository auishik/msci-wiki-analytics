"""Custom exceptions for the msci-wiki-analytics application."""


class WikiAnalyticsError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return (
                f"{self.message} (caused by {type(self.cause).__name__}: {self.cause})"
            )
        return self.message


class WikipediaFetchError(WikiAnalyticsError):
    """Base exception for Wikipedia fetching errors."""

    pass


class WikipediaAPIError(WikipediaFetchError):
    """Raised when Wikipedia API request fails.

    Covers connection errors, timeouts, rate limiting, and HTTP errors.
    """

    def __init__(
        self,
        message: str = "Wikipedia API error",
        status_code: int | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, cause)
        self.status_code = status_code


class RecoverableAPIError(WikipediaAPIError):
    """Raised for transient API errors that can be retried.

    Includes 429 rate limiting, timeouts, and connection errors.
    """

    def __init__(
        self,
        message: str = "Recoverable API error",
        status_code: int | None = None,
        retry_after: float | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, status_code, cause)
        self.retry_after = retry_after


class WikipediaParseError(WikipediaFetchError):
    """Raised when unable to parse Wikipedia API response."""

    def __init__(
        self,
        message: str = "Failed to parse Wikipedia response",
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, cause)
