"""Logging configuration for the application."""

import logging
import sys

from src.config import get_settings

LOG_FORMAT = "%(levelname)s | %(asctime)s | %(funcName)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """
    Configure logging for the application.

    Logs to stdout with format: LEVEL | TIME | FUNCTION | MESSAGE

    The log level is configured via the LOG_LEVEL environment variable.
    """
    settings = get_settings()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level.upper())
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
