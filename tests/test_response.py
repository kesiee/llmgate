"""Tests for LLMResponse dataclass."""

from llmgate.response import LLMResponse


class TestLLMResponse:
    def test_basic_fields(self):
        r = LLMResponse(
            text="Hello",
            model="gpt-4o",
            provider="openai",
            tokens_used=42,
            finish_reason="stop",
            raw={"id": "123"},
        )
        assert r.text == "Hello"
        assert r.model == "gpt-4o"
        assert r.provider == "openai"
        assert r.tokens_used == 42
        assert r.finish_reason == "stop"
        assert r.raw == {"id": "123"}

    def test_optional_tokens(self):
        r = LLMResponse(text="Hi", model="m", provider="p", tokens_used=None, finish_reason=None, raw={})
        assert r.tokens_used is None
        assert r.finish_reason is None

    def test_raw_preserved(self):
        raw = {"choices": [{"message": {"content": "hi"}}], "usage": {"total_tokens": 10}}
        r = LLMResponse(text="hi", model="m", provider="p", tokens_used=10, finish_reason="stop", raw=raw)
        assert r.raw["usage"]["total_tokens"] == 10

    def test_str(self):
        r = LLMResponse(text="Hello world", model="gpt-4o", provider="openai", tokens_used=5, finish_reason="stop", raw={})
        assert str(r) == "Hello world"

    def test_equality(self):
        kwargs = dict(text="a", model="m", provider="p", tokens_used=1, finish_reason="stop", raw={})
        assert LLMResponse(**kwargs) == LLMResponse(**kwargs)
