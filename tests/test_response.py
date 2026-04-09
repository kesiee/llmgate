"""Tests for LLMResponse and EmbeddingResponse dataclasses."""

from llmgate.response import EmbeddingResponse, LLMResponse, TokenUsage, ToolCall


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

    def test_tool_calls_default_empty(self):
        r = LLMResponse(text="", model="m", provider="p", tokens_used=None, finish_reason=None, raw={})
        assert r.tool_calls == []

    def test_tool_calls(self):
        tc = ToolCall(id="tc1", function="get_weather", arguments={"city": "NYC"})
        r = LLMResponse(text="", model="m", provider="p", tokens_used=None, finish_reason=None, raw={}, tool_calls=[tc])
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].function == "get_weather"
        assert r.tool_calls[0].arguments == {"city": "NYC"}

    def test_parsed_default_none(self):
        r = LLMResponse(text="", model="m", provider="p", tokens_used=None, finish_reason=None, raw={})
        assert r.parsed is None

    def test_parsed_set(self):
        r = LLMResponse(text='{"name":"test"}', model="m", provider="p", tokens_used=None, finish_reason=None, raw={})
        r.parsed = {"name": "test"}
        assert r.parsed == {"name": "test"}


class TestEmbeddingResponse:
    def test_basic(self):
        r = EmbeddingResponse(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            model="text-embedding-3-small",
            provider="openai",
            usage=TokenUsage(prompt_tokens=5, total_tokens=5),
        )
        assert len(r.embeddings) == 2
        assert r.model == "text-embedding-3-small"
        assert r.usage.prompt_tokens == 5

    def test_raw_default(self):
        r = EmbeddingResponse(embeddings=[], model="m", provider="p", usage=TokenUsage())
        assert r.raw == {}


class TestToolCall:
    def test_fields(self):
        tc = ToolCall(id="123", function="search", arguments={"q": "hello"})
        assert tc.id == "123"
        assert tc.function == "search"
        assert tc.arguments == {"q": "hello"}


class TestTokenUsage:
    def test_defaults(self):
        u = TokenUsage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0
