"""Rate-limiting middleware."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Callable

from llmgate.middleware.base import BaseMiddleware


class RateLimitMiddleware(BaseMiddleware):
    """Limits requests per minute using a sliding window.

    Args:
        rpm: Maximum requests per minute.
    """

    def __init__(self, rpm: int = 60) -> None:
        self.rpm = rpm
        self._window: deque[float] = deque()

    def _wait_if_needed(self) -> None:
        now = time.monotonic()
        # Purge entries older than 60s
        while self._window and now - self._window[0] > 60:
            self._window.popleft()
        if len(self._window) >= self.rpm:
            sleep_time = 60 - (now - self._window[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._window.append(time.monotonic())

    async def _await_if_needed(self) -> None:
        now = time.monotonic()
        while self._window and now - self._window[0] > 60:
            self._window.popleft()
        if len(self._window) >= self.rpm:
            sleep_time = 60 - (now - self._window[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        self._window.append(time.monotonic())

    def handle(self, messages: list[dict], call_next: Callable, **kwargs: Any) -> Any:
        self._wait_if_needed()
        return call_next(messages, **kwargs)

    async def ahandle(self, messages: list[dict], call_next: Callable, **kwargs: Any) -> Any:
        await self._await_if_needed()
        return await call_next(messages, **kwargs)
