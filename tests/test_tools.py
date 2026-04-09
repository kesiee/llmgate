"""Tests for tool/function calling support."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest

from llmgate.response import LLMResponse, ToolCall


def _make_httpx_response(data: dict, status: int = 200) -> httpx.Response:
    request = httpx.Request("POST", "https://fake")
    return httpx.Response(status, json=data, request=request)


def _patch_client(monkeypatch, response_data: dict):
    resp = _make_httpx_response(response_data)
    mock_client = MagicMock()
    mock_client.post = MagicMock(return_value=resp)
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(httpx, "Client", MagicMock(return_value=mock_client))
    return mock_client


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


class TestOpenAIToolCalling:
    @pytest.fixture()
    def provider(self):
        from llmgate.providers.openai import OpenAIProvider
        return OpenAIProvider({"provider": "openai", "model": "gpt-4o", "api_key": "k"})

    def test_tools_in_payload(self, provider, monkeypatch):
        data = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {"total_tokens": 5},
        }
        client = _patch_client(monkeypatch, data)
        provider.send([{"role": "user", "content": "weather?"}], tools=[WEATHER_TOOL])
        payload = client.post.call_args[1]["json"]
        assert "tools" in payload
        assert payload["tools"][0]["function"]["name"] == "get_weather"

    def test_tool_choice_in_payload(self, provider, monkeypatch):
        data = {
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {},
        }
        client = _patch_client(monkeypatch, data)
        provider.send(
            [{"role": "user", "content": "weather?"}],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
        )
        payload = client.post.call_args[1]["json"]
        assert payload["tool_choice"] == "auto"

    def test_parse_tool_calls_in_response(self, provider, monkeypatch):
        data = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "model": "gpt-4o",
            "usage": {"total_tokens": 20},
        }
        _patch_client(monkeypatch, data)
        result = provider.send([{"role": "user", "content": "weather NYC?"}], tools=[WEATHER_TOOL])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].function == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "NYC"}

    def test_no_tool_calls_returns_empty_list(self, provider, monkeypatch):
        data = {
            "choices": [{"message": {"content": "no tools"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {"total_tokens": 5},
        }
        _patch_client(monkeypatch, data)
        result = provider.send([{"role": "user", "content": "hi"}])
        assert result.tool_calls == []


class TestAnthropicToolCalling:
    @pytest.fixture()
    def provider(self):
        from llmgate.providers.anthropic import AnthropicProvider
        return AnthropicProvider({"provider": "anthropic", "model": "claude-sonnet-4-20250514", "api_key": "k"})

    def test_tools_converted_to_anthropic_format(self, provider, monkeypatch):
        data = {
            "content": [{"type": "text", "text": "ok"}],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        client = _patch_client(monkeypatch, data)
        provider.send(
            [{"role": "user", "content": "weather?"}],
            tools=[WEATHER_TOOL],
        )
        payload = client.post.call_args[1]["json"]
        assert "tools" in payload
        assert payload["tools"][0]["name"] == "get_weather"
        assert "input_schema" in payload["tools"][0]

    def test_parse_tool_use_response(self, provider, monkeypatch):
        data = {
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "tu_456",
                    "name": "get_weather",
                    "input": {"city": "London"},
                },
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 8},
        }
        _patch_client(monkeypatch, data)
        result = provider.send([{"role": "user", "content": "weather London?"}], tools=[WEATHER_TOOL])
        assert result.text == "Let me check."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "tu_456"
        assert result.tool_calls[0].function == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "London"}


class TestGeminiToolCalling:
    @pytest.fixture()
    def provider(self):
        from llmgate.providers.gemini import GeminiProvider
        return GeminiProvider({"provider": "gemini", "model": "gemini-2.0-flash", "api_key": "k"})

    def test_tools_in_payload(self, provider, monkeypatch):
        data = {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
            "usageMetadata": {},
        }
        client = _patch_client(monkeypatch, data)
        provider.send([{"role": "user", "content": "hi"}], tools=[WEATHER_TOOL])
        payload = client.post.call_args[1]["json"]
        assert "tools" in payload

    def test_parse_function_call_response(self, provider, monkeypatch):
        data = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "Paris"},
                        },
                    }],
                },
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
        }
        _patch_client(monkeypatch, data)
        result = provider.send([{"role": "user", "content": "weather Paris?"}], tools=[WEATHER_TOOL])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "Paris"}


class TestStructuredOutput:
    def test_structured_output_via_gate(self, tmp_path):
        from llmgate.gate import LLMGate, _apply_structured_output
        from pydantic import BaseModel

        class Movie(BaseModel):
            title: str
            year: int

        resp = LLMResponse(
            text='{"title": "Inception", "year": 2010}',
            model="m", provider="p", tokens_used=10, finish_reason="stop", raw={},
        )
        result = _apply_structured_output(resp, Movie)
        assert result.parsed is not None
        assert result.parsed.title == "Inception"
        assert result.parsed.year == 2010

    def test_structured_output_from_code_block(self):
        from llmgate.gate import _apply_structured_output
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str

        resp = LLMResponse(
            text='```json\n{"name": "widget"}\n```',
            model="m", provider="p", tokens_used=5, finish_reason="stop", raw={},
        )
        result = _apply_structured_output(resp, Item)
        assert result.parsed is not None
        assert result.parsed.name == "widget"

    def test_structured_output_invalid_json(self):
        from llmgate.gate import _apply_structured_output
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str

        resp = LLMResponse(
            text="not json at all",
            model="m", provider="p", tokens_used=5, finish_reason="stop", raw={},
        )
        result = _apply_structured_output(resp, Item)
        assert result.parsed is None

    def test_dict_format_skips_parsing(self):
        from llmgate.gate import _apply_structured_output

        resp = LLMResponse(
            text='{"key": "val"}',
            model="m", provider="p", tokens_used=5, finish_reason="stop", raw={},
        )
        result = _apply_structured_output(resp, {"type": "json_object"})
        assert result.parsed is None
