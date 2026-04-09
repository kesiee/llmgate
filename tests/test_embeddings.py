"""Tests for embeddings support."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from llmgate.exceptions import EmbeddingsNotSupported
from llmgate.response import EmbeddingResponse


def _make_httpx_response(data, status: int = 200) -> httpx.Response:
    request = httpx.Request("POST", "https://fake")
    return httpx.Response(status, json=data, request=request)


def _patch_client(monkeypatch, response_data):
    resp = _make_httpx_response(response_data)
    mock_client = MagicMock()
    mock_client.post = MagicMock(return_value=resp)
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(httpx, "Client", MagicMock(return_value=mock_client))
    return mock_client


class TestOpenAIEmbeddings:
    @pytest.fixture()
    def provider(self):
        from llmgate.providers.openai import OpenAIProvider
        return OpenAIProvider({"provider": "openai", "model": "text-embedding-3-small", "api_key": "k"})

    def test_embed_single(self, provider, monkeypatch):
        data = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }
        _patch_client(monkeypatch, data)
        result = provider.embed("hello")
        assert isinstance(result, EmbeddingResponse)
        assert len(result.embeddings) == 1
        assert result.embeddings[0] == [0.1, 0.2, 0.3]
        assert result.usage.prompt_tokens == 3

    def test_embed_batch(self, provider, monkeypatch):
        data = {
            "data": [{"embedding": [0.1]}, {"embedding": [0.2]}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        _patch_client(monkeypatch, data)
        result = provider.embed(["hello", "world"])
        assert len(result.embeddings) == 2

    def test_embed_url(self, provider, monkeypatch):
        data = {"data": [{"embedding": [0.1]}], "model": "m", "usage": {"prompt_tokens": 1, "total_tokens": 1}}
        client = _patch_client(monkeypatch, data)
        provider.embed("hi")
        url = client.post.call_args[0][0]
        assert url.endswith("/embeddings")

    def test_embed_dimensions(self, provider, monkeypatch):
        data = {"data": [{"embedding": [0.1]}], "model": "m", "usage": {"prompt_tokens": 1, "total_tokens": 1}}
        client = _patch_client(monkeypatch, data)
        provider.embed("hi", dimensions=256)
        payload = client.post.call_args[1]["json"]
        assert payload["dimensions"] == 256


class TestGeminiEmbeddings:
    @pytest.fixture()
    def provider(self):
        from llmgate.providers.gemini import GeminiProvider
        return GeminiProvider({"provider": "gemini", "model": "text-embedding-004", "api_key": "k"})

    def test_embed(self, provider, monkeypatch):
        data = {"embeddings": [{"values": [0.5, 0.6]}]}
        _patch_client(monkeypatch, data)
        result = provider.embed("hello")
        assert isinstance(result, EmbeddingResponse)
        assert result.embeddings[0] == [0.5, 0.6]


class TestCohereEmbeddings:
    @pytest.fixture()
    def provider(self):
        from llmgate.providers.cohere import CohereProvider
        return CohereProvider({"provider": "cohere", "model": "embed-english-v3.0", "api_key": "k"})

    def test_embed(self, provider, monkeypatch):
        data = {
            "embeddings": {"float": [[0.1, 0.2]]},
            "meta": {"billed_units": {"input_tokens": 3}},
        }
        _patch_client(monkeypatch, data)
        result = provider.embed("hello")
        assert isinstance(result, EmbeddingResponse)
        assert result.embeddings[0] == [0.1, 0.2]


class TestUnsupportedEmbeddings:
    def test_anthropic_raises(self):
        from llmgate.providers.anthropic import AnthropicProvider
        p = AnthropicProvider({"provider": "anthropic", "model": "claude-sonnet-4-20250514", "api_key": "k"})
        with pytest.raises(EmbeddingsNotSupported):
            p.embed("hello")

    def test_ollama_raises(self):
        from llmgate.providers.ollama import OllamaProvider
        p = OllamaProvider({"provider": "ollama", "model": "llama3"})
        with pytest.raises(EmbeddingsNotSupported):
            p.embed("hello")


class TestGateEmbeddings:
    def test_embed_delegates(self, tmp_path):
        from llmgate.gate import LLMGate
        cfg = tmp_path / "llmgate.yaml"
        cfg.write_text("provider: openai\nmodel: text-embedding-3-small\napi_key: k\n")
        gate = LLMGate(config_path=str(cfg))
        fake = EmbeddingResponse(embeddings=[[0.1]], model="m", provider="p", usage=MagicMock())
        gate._provider.embed = MagicMock(return_value=fake)
        result = gate.embed("hello")
        assert result is fake

    def test_embed_not_supported(self, tmp_path):
        from llmgate.gate import LLMGate
        cfg = tmp_path / "llmgate.yaml"
        cfg.write_text("provider: anthropic\nmodel: claude-sonnet-4-20250514\napi_key: k\n")
        gate = LLMGate(config_path=str(cfg))
        with pytest.raises(EmbeddingsNotSupported):
            gate.embed("hello")
