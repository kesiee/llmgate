"""Base middleware interface."""

from __future__ import annotations

from typing import Any, Callable


class BaseMiddleware:
    """Middleware that wraps send/stream calls.

    Subclass and override ``handle`` / ``ahandle`` for request/response
    middleware, or ``stream_handle`` / ``astream_handle`` for streaming.
    """

    def handle(self, messages: list[dict], call_next: Callable, **kwargs: Any) -> Any:
        """Sync middleware hook. Call ``call_next(messages, **kwargs)`` to proceed."""
        return call_next(messages, **kwargs)

    async def ahandle(self, messages: list[dict], call_next: Callable, **kwargs: Any) -> Any:
        """Async middleware hook."""
        return await call_next(messages, **kwargs)

    def stream_handle(self, messages: list[dict], call_next: Callable, **kwargs: Any) -> Any:
        """Sync streaming middleware hook."""
        return call_next(messages, **kwargs)

    async def astream_handle(self, messages: list[dict], call_next: Callable, **kwargs: Any) -> Any:
        """Async streaming middleware hook."""
        return call_next(messages, **kwargs)
