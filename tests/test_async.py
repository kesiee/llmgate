"""Tests for async methods across providers and gate."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llmgate.response import LLMResponse


def _make_httpx_response(data, status: int = 200) -> httpx.Response:
    request = httpx.Request("POST", "https://fake")
    return httpx.Response(status, json=data, request=request)


def _fake_response(**overrides):
    defaults = dict(text="ok", model="m", provider="p", tokens_used=5, finish_reason="stop", raw={})
    defaults.update(overrides)
    return LLMResponse(**defaults)


class TestOpenAIAsync:
    @pytest.fixture()
    def provider(self):
        from llmgate.providers.openai import OpenAIProvider
        return OpenAIProvider({"provider": "openai", "model": "gpt-4o", "api_key": "sk-test"})

    @pytest.mark.anyio
    async def test_asend(self, provider, monkeypatch):
        data = {
            "choices": [{"message": {"content": "async hi"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {"total_tokens": 10},
        }
        resp = _make_httpx_response(data)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))

        result = await provider.asend([{"role": "user", "content": "hi"}])
        assert result.text == "async hi"
        assert result.provider == "openai"

    @pytest.mark.anyio
    async def test_astream(self, provider, monkeypatch):
        lines = [
            'data: {"choices":[{"delta":{"content":"a"}}]}',
            'data: {"choices":[{"delta":{"content":"b"}}]}',
            "data: [DONE]",
        ]

        mock_resp = AsyncMock()
        mock_resp.is_success = True

        async def fake_aiter_lines():
            for line in lines:
                yield line

        mock_resp.aiter_lines = fake_aiter_lines

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncContextManager(mock_resp))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))

        chunks = []
        async for chunk in provider.astream([{"role": "user", "content": "hi"}]):
            chunks.append(chunk)
        assert chunks == ["a", "b"]


class TestAnthropicAsync:
    @pytest.mark.anyio
    async def test_asend(self, monkeypatch):
        from llmgate.providers.anthropic import AnthropicProvider
        p = AnthropicProvider({"provider": "anthropic", "model": "claude-sonnet-4-20250514", "api_key": "k"})
        data = {
            "content": [{"type": "text", "text": "async claude"}],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
        resp = _make_httpx_response(data)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))

        result = await p.asend([{"role": "user", "content": "hi"}])
        assert result.text == "async claude"


class TestGeminiAsync:
    @pytest.mark.anyio
    async def test_asend(self, monkeypatch):
        from llmgate.providers.gemini import GeminiProvider
        p = GeminiProvider({"provider": "gemini", "model": "gemini-2.0-flash", "api_key": "k"})
        data = {
            "candidates": [{"content": {"parts": [{"text": "async gemini"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2},
        }
        resp = _make_httpx_response(data)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))

        result = await p.asend([{"role": "user", "content": "hi"}])
        assert result.text == "async gemini"


class TestOllamaAsync:
    @pytest.mark.anyio
    async def test_asend(self, monkeypatch):
        from llmgate.providers.ollama import OllamaProvider
        p = OllamaProvider({"provider": "ollama", "model": "llama3"})
        data = {"message": {"content": "async ollama"}, "model": "llama3", "eval_count": 10}
        resp = _make_httpx_response(data)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))

        result = await p.asend([{"role": "user", "content": "hi"}])
        assert result.text == "async ollama"


class TestGateAsync:
    @pytest.fixture()
    def gate(self, tmp_path):
        from llmgate.gate import LLMGate
        cfg = tmp_path / "llmgate.yaml"
        cfg.write_text("provider: openai\nmodel: gpt-4o\napi_key: sk-test\n")
        return LLMGate(config_path=str(cfg))

    @pytest.mark.anyio
    async def test_achat(self, gate):
        fake = _fake_response(text="async gate")
        gate._provider.asend = AsyncMock(return_value=fake)
        result = await gate.achat("hi")
        assert result.text == "async gate"

    @pytest.mark.anyio
    async def test_achat_messages(self, gate):
        fake = _fake_response(text="async msgs")
        gate._provider.asend = AsyncMock(return_value=fake)
        result = await gate.achat_messages([{"role": "user", "content": "hi"}])
        assert result.text == "async msgs"

    @pytest.mark.anyio
    async def test_astream(self, gate):
        async def fake_astream(messages, **kwargs):
            for chunk in ["a", "b", "c"]:
                yield chunk

        gate._provider.astream = fake_astream
        chunks = []
        async for chunk in gate.astream("hi"):
            chunks.append(chunk)
        assert chunks == ["a", "b", "c"]

    @pytest.mark.anyio
    async def test_astream_messages(self, gate):
        async def fake_astream(messages, **kwargs):
            yield "x"

        gate._provider.astream = fake_astream
        chunks = []
        async for chunk in gate.astream_messages([{"role": "user", "content": "hi"}]):
            chunks.append(chunk)
        assert chunks == ["x"]


# Helper for async context manager mocking
class AsyncContextManager:
    def __init__(self, obj):
        self.obj = obj

    async def __aenter__(self):
        return self.obj

    async def __aexit__(self, *args):
        return False
