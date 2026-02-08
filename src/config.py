from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(extra="ignore")

    # Wikipedia API settings
    wiki_user_agent: str = (
        "msci-wiki-analytics/0.1.0 "
        "(https://github.com/user/msci-wiki-analytics; contact@example.com)"
    )

    # Rate limiting settings
    max_concurrent_requests: int = 10
    max_requests_per_second: float = 5.0

    # HTTP client settings
    request_timeout: float = 30.0

    # Retry settings
    retry_max_attempts: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 30.0
    retry_multiplier: float = 2.0
    retry_jitter_max: float = 1.0


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
