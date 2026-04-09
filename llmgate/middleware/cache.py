"""Simple in-memory TTL cache middleware."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable

from llmgate.middleware.base import BaseMiddleware


class CacheMiddleware(BaseMiddleware):
    """Caches responses by message content with a TTL.

    Args:
        ttl: Time-to-live in seconds for cached entries.
    """

    def __init__(self, ttl: int = 300) -> None:
        self.ttl = ttl
        self._cache: dict[str, tuple[float, Any]] = {}

    def _key(self, messages: list[dict], **kwargs: Any) -> str:
        raw = json.dumps({"messages": messages, "kwargs": kwargs}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _get(self, key: str) -> Any | None:
        if key in self._cache:
            ts, value = self._cache[key]
            if time.monotonic() - ts < self.ttl:
                return value
            del self._cache[key]
        return None

    def _set(self, key: str, value: Any) -> None:
        self._cache[key] = (time.monotonic(), value)

    def handle(self, messages: list[dict], call_next: Callable, **kwargs: Any) -> Any:
        key = self._key(messages, **kwargs)
        cached = self._get(key)
        if cached is not None:
            return cached
        result = call_next(messages, **kwargs)
        self._set(key, result)
        return result

    async def ahandle(self, messages: list[dict], call_next: Callable, **kwargs: Any) -> Any:
        key = self._key(messages, **kwargs)
        cached = self._get(key)
        if cached is not None:
            return cached
        result = await call_next(messages, **kwargs)
        self._set(key, result)
        return result
