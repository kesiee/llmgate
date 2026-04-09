"""Tests for middleware system."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from llmgate.middleware import (
    BaseMiddleware,
    CacheMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
    RetryMiddleware,
)
from llmgate.exceptions import RateLimitError
from llmgate.response import LLMResponse


def _fake_response(**overrides):
    defaults = dict(text="ok", model="m", provider="p", tokens_used=5, finish_reason="stop", raw={})
    defaults.update(overrides)
    return LLMResponse(**defaults)


class TestBaseMiddleware:
    def test_passthrough(self):
        mw = BaseMiddleware()
        call_next = MagicMock(return_value=_fake_response())
        result = mw.handle([{"role": "user", "content": "hi"}], call_next)
        assert result.text == "ok"
        call_next.assert_called_once()


class TestRetryMiddleware:
    def test_no_retry_on_success(self):
        mw = RetryMiddleware(max_retries=3)
        call_next = MagicMock(return_value=_fake_response())
        result = mw.handle([], call_next)
        assert call_next.call_count == 1
        assert result.text == "ok"

    def test_retries_on_rate_limit(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        mw = RetryMiddleware(max_retries=2, backoff_factor=0.01)
        call_next = MagicMock(side_effect=[
            RateLimitError("test"),
            RateLimitError("test"),
            _fake_response(),
        ])
        result = mw.handle([], call_next)
        assert call_next.call_count == 3
        assert result.text == "ok"

    def test_raises_after_max_retries(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        mw = RetryMiddleware(max_retries=2, backoff_factor=0.01)
        call_next = MagicMock(side_effect=RateLimitError("test"))
        with pytest.raises(RateLimitError):
            mw.handle([], call_next)
        assert call_next.call_count == 3  # initial + 2 retries

    def test_no_retry_on_non_retryable(self):
        mw = RetryMiddleware(max_retries=3)
        call_next = MagicMock(side_effect=ValueError("bad"))
        with pytest.raises(ValueError):
            mw.handle([], call_next)
        assert call_next.call_count == 1


class TestCacheMiddleware:
    def test_caches_identical_requests(self):
        mw = CacheMiddleware(ttl=60)
        call_next = MagicMock(return_value=_fake_response(text="first"))
        msgs = [{"role": "user", "content": "hi"}]

        result1 = mw.handle(msgs, call_next)
        result2 = mw.handle(msgs, call_next)

        assert result1.text == "first"
        assert result2.text == "first"
        assert call_next.call_count == 1  # second call was cached

    def test_different_messages_not_cached(self):
        mw = CacheMiddleware(ttl=60)
        call_next = MagicMock(side_effect=[_fake_response(text="a"), _fake_response(text="b")])

        r1 = mw.handle([{"role": "user", "content": "hi"}], call_next)
        r2 = mw.handle([{"role": "user", "content": "bye"}], call_next)

        assert r1.text == "a"
        assert r2.text == "b"
        assert call_next.call_count == 2

    def test_ttl_expiry(self, monkeypatch):
        mw = CacheMiddleware(ttl=1)
        call_next = MagicMock(side_effect=[_fake_response(text="old"), _fake_response(text="new")])
        msgs = [{"role": "user", "content": "hi"}]

        mw.handle(msgs, call_next)

        # Simulate TTL expiry by manipulating the cache timestamp
        key = list(mw._cache.keys())[0]
        mw._cache[key] = (time.monotonic() - 2, mw._cache[key][1])

        result = mw.handle(msgs, call_next)
        assert result.text == "new"
        assert call_next.call_count == 2


class TestLoggingMiddleware:
    def test_logs_and_passes_through(self):
        mw = LoggingMiddleware(level="DEBUG")
        call_next = MagicMock(return_value=_fake_response())
        result = mw.handle([{"role": "user", "content": "hi"}], call_next)
        assert result.text == "ok"
        call_next.assert_called_once()


class TestRateLimitMiddleware:
    def test_allows_within_limit(self):
        mw = RateLimitMiddleware(rpm=10)
        call_next = MagicMock(return_value=_fake_response())
        for _ in range(5):
            mw.handle([], call_next)
        assert call_next.call_count == 5

    def test_window_tracking(self):
        mw = RateLimitMiddleware(rpm=100)
        call_next = MagicMock(return_value=_fake_response())
        mw.handle([], call_next)
        assert len(mw._window) == 1


class TestMiddlewareChaining:
    def test_multiple_middleware(self):
        """Middleware should compose: cache wraps logging wraps call."""
        cache = CacheMiddleware(ttl=60)
        log = LoggingMiddleware(level="DEBUG")
        call_next = MagicMock(return_value=_fake_response(text="chained"))
        msgs = [{"role": "user", "content": "hi"}]

        # Build chain manually: cache -> log -> call_next
        def chain(msgs, **kw):
            return cache.handle(msgs, lambda m, **k: log.handle(m, call_next, **k), **kw)

        result = chain(msgs)
        assert result.text == "chained"
