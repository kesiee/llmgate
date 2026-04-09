"""Retry middleware with exponential backoff."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from llmgate.exceptions import RateLimitError
from llmgate.middleware.base import BaseMiddleware


class RetryMiddleware(BaseMiddleware):
    """Retries on transient errors with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        backoff_factor: Multiplier for exponential delay (delay = factor * 2^attempt).
        retryable: Exception types to retry on.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        retryable: tuple[type[Exception], ...] = (RateLimitError, ConnectionError, TimeoutError),
    ) -> None:
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.retryable = retryable

    def handle(self, messages: list[dict], call_next: Callable, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return call_next(messages, **kwargs)
            except self.retryable as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_factor * (2 ** attempt))
        raise last_exc  # type: ignore[misc]

    async def ahandle(self, messages: list[dict], call_next: Callable, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await call_next(messages, **kwargs)
            except self.retryable as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(self.backoff_factor * (2 ** attempt))
        raise last_exc  # type: ignore[misc]
