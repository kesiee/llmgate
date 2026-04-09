"""Logging middleware."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from llmgate.middleware.base import BaseMiddleware

logger = logging.getLogger("llmgate")


class LoggingMiddleware(BaseMiddleware):
    """Logs request/response details.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, etc.).
    """

    def __init__(self, level: str = "INFO") -> None:
        self.level = getattr(logging, level.upper(), logging.INFO)

    def handle(self, messages: list[dict], call_next: Callable, **kwargs: Any) -> Any:
        logger.log(self.level, "llmgate request: %d message(s)", len(messages))
        start = time.monotonic()
        result = call_next(messages, **kwargs)
        elapsed = time.monotonic() - start
        logger.log(
            self.level,
            "llmgate response: provider=%s model=%s tokens=%s time=%.2fs",
            result.provider, result.model, result.tokens_used, elapsed,
        )
        return result

    async def ahandle(self, messages: list[dict], call_next: Callable, **kwargs: Any) -> Any:
        logger.log(self.level, "llmgate async request: %d message(s)", len(messages))
        start = time.monotonic()
        result = await call_next(messages, **kwargs)
        elapsed = time.monotonic() - start
        logger.log(
            self.level,
            "llmgate async response: provider=%s model=%s tokens=%s time=%.2fs",
            result.provider, result.model, result.tokens_used, elapsed,
        )
        return result
