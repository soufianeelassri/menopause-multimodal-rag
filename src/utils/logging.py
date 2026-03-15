"""Structured logging configuration for the MenoGuide MRAG system.

Provides consistent, structured logging across all modules using structlog.
Outputs colored console logs in development, JSON in production.
"""

from __future__ import annotations

import logging
import sys

import structlog

from src.config.settings import get_settings


def _configure_structlog(log_level: str = "INFO") -> None:
    """Configure structlog processors and output format.

    Args:
        log_level: Minimum log level to emit (DEBUG, INFO, WARNING, ERROR).
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))


_configured = False


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named structured logger.

    Configures structlog on first call using settings from the environment.

    Args:
        name: Logger name, typically the module name.

    Returns:
        A bound structlog logger instance.
    """
    global _configured
    if not _configured:
        try:
            log_level = get_settings().log_level
        except Exception:
            log_level = "INFO"
        _configure_structlog(log_level)
        _configured = True

    return structlog.get_logger(name)
