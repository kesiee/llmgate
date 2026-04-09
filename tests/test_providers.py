"""Tests for LLM providers — send/stream with mocked HTTP."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from llmgate.response import LLMResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_httpx_response(data: dict, status: int = 200) -> httpx.Response:
    """Build a real httpx.Response from a dict."""
    request = httpx.Request("POST", "https://fake")
    return httpx.Response(status, json=data, request=request)


def _make_stream_response(lines: list[str], status: int = 200) -> MagicMock:
    """Build a mock streaming response that yields SSE lines."""
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.iter_lines = MagicMock(return_value=iter(lines))
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def _patch_client(monkeypatch, provider, response_data: dict):
    """Patch httpx.Client so provider.send() returns canned data."""
    resp = _make_httpx_response(response_data)

    mock_client = MagicMock()
    mock_client.post = MagicMock(return_value=resp)
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    monkeypatch.setattr(httpx, "Client", MagicMock(return_value=mock_client))
    return mock_client


def _patch_client_stream(monkeypatch, lines: list[str]):
    """Patch httpx.Client so provider.stream() yields canned SSE lines."""
    stream_resp = _make_stream_response(lines)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=stream_resp)
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    monkeypatch.setattr(httpx, "Client", MagicMock(return_value=mock_client))
    return mock_client


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class TestOpenAIProvider:
    @pytest.fixture()
    def provider(self):
        from llmgate.providers.openai import OpenAIProvider
        return OpenAIProvider({"provider": "openai", "model": "gpt-4o", "api_key": "sk-test"})

    def test_send(self, provider, monkeypatch):
        data = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {"total_tokens": 15},
        }
        _patch_client(monkeypatch, provider, data)
        result = provider.send([{"role": "user", "content": "hello"}])
        assert isinstance(result, LLMResponse)
        assert result.text == "hi"
        assert result.model == "gpt-4o"
        assert result.provider == "openai"
        assert result.tokens_used == 15
        assert result.finish_reason == "stop"

    def test_send_passes_config_params(self, provider, monkeypatch):
        provider.config["temperature"] = 0.5
        data = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {"total_tokens": 5},
        }
        client = _patch_client(monkeypatch, provider, data)
        provider.send([{"role": "user", "content": "hi"}])
        payload = client.post.call_args[1]["json"]
        assert payload["temperature"] == 0.5

    def test_send_kwargs_override_config(self, provider, monkeypatch):
        provider.config["temperature"] = 0.5
        data = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {},
        }
        client = _patch_client(monkeypatch, provider, data)
        provider.send([{"role": "user", "content": "hi"}], temperature=0.9)
        payload = client.post.call_args[1]["json"]
        assert payload["temperature"] == 0.9

    def test_stream(self, provider, monkeypatch):
        lines = [
            'data: {"choices":[{"delta":{"content":"hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            "data: [DONE]",
        ]
        _patch_client_stream(monkeypatch, lines)
        chunks = list(provider.stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["hel", "lo"]

    def test_custom_base_url(self):
        from llmgate.providers.openai import OpenAIProvider
        p = OpenAIProvider({"provider": "openai", "model": "m", "base_url": "http://my-proxy.com/v1"})
        assert p._get_url() == "http://my-proxy.com/v1/chat/completions"

    def test_headers_include_bearer(self, provider):
        headers = provider._get_headers()
        assert headers["Authorization"] == "Bearer sk-test"


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class TestAnthropicProvider:
    @pytest.fixture()
    def provider(self):
        from llmgate.providers.anthropic import AnthropicProvider
        return AnthropicProvider({"provider": "anthropic", "model": "claude-sonnet-4-20250514", "api_key": "sk-ant-test"})

    def test_send(self, provider, monkeypatch):
        data = {
            "content": [{"text": "hi there"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        _patch_client(monkeypatch, provider, data)
        result = provider.send([{"role": "user", "content": "hello"}])
        assert result.text == "hi there"
        assert result.provider == "anthropic"
        assert result.tokens_used == 15

    def test_system_message_extracted(self, provider, monkeypatch):
        data = {
            "content": [{"text": "ok"}],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        client = _patch_client(monkeypatch, provider, data)
        provider.send([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ])
        payload = client.post.call_args[1]["json"]
        assert payload["system"] == "be brief"
        assert all(m["role"] != "system" for m in payload["messages"])

    def test_headers(self, provider):
        headers = provider._get_headers()
        assert headers["x-api-key"] == "sk-ant-test"
        assert "anthropic-version" in headers

    def test_stream(self, provider, monkeypatch):
        lines = [
            'data: {"type":"content_block_start"}',
            'data: {"type":"content_block_delta","delta":{"text":"he"}}',
            'data: {"type":"content_block_delta","delta":{"text":"y"}}',
            'data: {"type":"message_stop"}',
        ]
        _patch_client_stream(monkeypatch, lines)
        chunks = list(provider.stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["he", "y"]


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

class TestGeminiProvider:
    @pytest.fixture()
    def provider(self):
        from llmgate.providers.gemini import GeminiProvider
        return GeminiProvider({"provider": "gemini", "model": "gemini-2.0-flash", "api_key": "gk-test"})

    def test_send(self, provider, monkeypatch):
        data = {
            "candidates": [{"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
        }
        _patch_client(monkeypatch, provider, data)
        result = provider.send([{"role": "user", "content": "hi"}])
        assert result.text == "hello"
        assert result.provider == "gemini"
        assert result.tokens_used == 8

    def test_message_conversion(self, provider):
        contents, system = provider._convert_messages([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ])
        assert system == "sys"
        assert len(contents) == 2
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"

    def test_url_includes_model_and_key(self, provider):
        url = provider._url()
        assert "gemini-2.0-flash" in url
        assert "key=gk-test" in url

    def test_stream(self, provider, monkeypatch):
        lines = [
            'data: {"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}',
        ]
        _patch_client_stream(monkeypatch, lines)
        chunks = list(provider.stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["ok"]


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class TestOllamaProvider:
    @pytest.fixture()
    def provider(self):
        from llmgate.providers.ollama import OllamaProvider
        return OllamaProvider({"provider": "ollama", "model": "llama3"})

    def test_send(self, provider, monkeypatch):
        data = {
            "message": {"content": "hey"},
            "model": "llama3",
            "eval_count": 20,
            "done_reason": "stop",
        }
        _patch_client(monkeypatch, provider, data)
        result = provider.send([{"role": "user", "content": "hi"}])
        assert result.text == "hey"
        assert result.provider == "ollama"
        assert result.tokens_used == 20

    def test_default_url(self, provider):
        assert provider._get_url() == "http://localhost:11434/api/chat"

    def test_custom_base_url(self):
        from llmgate.providers.ollama import OllamaProvider
        p = OllamaProvider({"provider": "ollama", "model": "m", "base_url": "http://remote:11434"})
        assert p._get_url() == "http://remote:11434/api/chat"

    def test_stream(self, provider, monkeypatch):
        lines = [
            json.dumps({"message": {"content": "a"}}),
            json.dumps({"message": {"content": "b"}}),
        ]
        _patch_client_stream(monkeypatch, lines)
        chunks = list(provider.stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["a", "b"]


# ---------------------------------------------------------------------------
# Cohere
# ---------------------------------------------------------------------------

class TestCohereProvider:
    @pytest.fixture()
    def provider(self):
        from llmgate.providers.cohere import CohereProvider
        return CohereProvider({"provider": "cohere", "model": "command-r-plus", "api_key": "co-test"})

    def test_send(self, provider, monkeypatch):
        data = {
            "message": {"content": [{"text": "hey"}]},
            "model": "command-r-plus",
            "usage": {"tokens": {"input_tokens": 10, "output_tokens": 5}},
            "finish_reason": "COMPLETE",
        }
        _patch_client(monkeypatch, provider, data)
        result = provider.send([{"role": "user", "content": "hi"}])
        assert result.text == "hey"
        assert result.provider == "cohere"
        assert result.tokens_used == 15

    def test_headers(self, provider):
        headers = provider._get_headers()
        assert headers["Authorization"] == "Bearer co-test"


# ---------------------------------------------------------------------------
# OpenAI-compatible thin wrappers
# ---------------------------------------------------------------------------

class TestOpenAICompatibleProviders:
    """All providers that simply subclass OpenAIProvider with a different BASE_URL."""

    @pytest.mark.parametrize("provider_name,module,classname,expected_url_fragment", [
        ("groq", "llmgate.providers.groq", "GroqProvider", "api.groq.com"),
        ("together", "llmgate.providers.together", "TogetherProvider", "api.together.xyz"),
        ("fireworks", "llmgate.providers.fireworks", "FireworksProvider", "api.fireworks.ai"),
        ("deepseek", "llmgate.providers.deepseek", "DeepSeekProvider", "api.deepseek.com"),
        ("perplexity", "llmgate.providers.perplexity", "PerplexityProvider", "api.perplexity.ai"),
        ("xai", "llmgate.providers.xai", "XAIProvider", "api.x.ai"),
        ("mistral", "llmgate.providers.mistral", "MistralProvider", "api.mistral.ai"),
        ("ai21", "llmgate.providers.ai21", "AI21Provider", "api.ai21.com"),
        ("lmstudio", "llmgate.providers.lmstudio", "LMStudioProvider", "localhost:1234"),
    ])
    def test_base_url_and_name(self, provider_name, module, classname, expected_url_fragment):
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, classname)
        p = cls({"provider": provider_name, "model": "test-model", "api_key": "k"})
        assert p.provider_name == provider_name
        assert expected_url_fragment in p._get_url()

    @pytest.mark.parametrize("provider_name,module,classname", [
        ("groq", "llmgate.providers.groq", "GroqProvider"),
        ("together", "llmgate.providers.together", "TogetherProvider"),
        ("fireworks", "llmgate.providers.fireworks", "FireworksProvider"),
        ("deepseek", "llmgate.providers.deepseek", "DeepSeekProvider"),
        ("perplexity", "llmgate.providers.perplexity", "PerplexityProvider"),
        ("xai", "llmgate.providers.xai", "XAIProvider"),
        ("mistral", "llmgate.providers.mistral", "MistralProvider"),
        ("ai21", "llmgate.providers.ai21", "AI21Provider"),
    ])
    def test_send_works_via_openai_base(self, provider_name, module, classname, monkeypatch):
        """Each thin wrapper should produce a valid LLMResponse via OpenAI send()."""
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, classname)
        p = cls({"provider": provider_name, "model": "m", "api_key": "k"})

        data = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "model": "m",
            "usage": {"total_tokens": 5},
        }
        _patch_client(monkeypatch, p, data)
        result = p.send([{"role": "user", "content": "hi"}])
        assert result.text == "ok"
        assert result.provider == provider_name


# ---------------------------------------------------------------------------
# OpenRouter (extra headers)
# ---------------------------------------------------------------------------

class TestOpenRouterProvider:
    def test_extra_headers(self):
        from llmgate.providers.openrouter import OpenRouterProvider
        p = OpenRouterProvider({
            "provider": "openrouter", "model": "m", "api_key": "k",
            "site_url": "https://myapp.com", "app_name": "MyApp",
        })
        headers = p._get_headers()
        assert headers["HTTP-Referer"] == "https://myapp.com"
        assert headers["X-Title"] == "MyApp"
        assert "Bearer k" in headers["Authorization"]

    def test_no_extra_headers_when_absent(self):
        from llmgate.providers.openrouter import OpenRouterProvider
        p = OpenRouterProvider({"provider": "openrouter", "model": "m", "api_key": "k"})
        headers = p._get_headers()
        assert "HTTP-Referer" not in headers
        assert "X-Title" not in headers


# ---------------------------------------------------------------------------
# Azure OpenAI
# ---------------------------------------------------------------------------

class TestAzureOpenAIProvider:
    @pytest.fixture()
    def provider(self):
        from llmgate.providers.azure_openai import AzureOpenAIProvider
        return AzureOpenAIProvider({
            "provider": "azure_openai", "model": "gpt-4o",
            "api_key": "az-key", "resource_name": "myresource",
            "deployment_name": "mydeployment",
        })

    def test_url_construction(self, provider):
        url = provider._get_url()
        assert "myresource.openai.azure.com" in url
        assert "mydeployment" in url
        assert "api-version=2024-02-01" in url

    def test_custom_api_version(self):
        from llmgate.providers.azure_openai import AzureOpenAIProvider
        p = AzureOpenAIProvider({
            "provider": "azure_openai", "model": "m", "api_key": "k",
            "resource_name": "r", "deployment_name": "d", "api_version": "2025-01-01",
        })
        assert "api-version=2025-01-01" in p._get_url()

    def test_headers_use_api_key_header(self, provider):
        headers = provider._get_headers()
        assert headers["api-key"] == "az-key"
        assert "Authorization" not in headers

    def test_send(self, provider, monkeypatch):
        data = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {"total_tokens": 12},
        }
        _patch_client(monkeypatch, provider, data)
        result = provider.send([{"role": "user", "content": "hi"}])
        assert result.text == "ok"
        assert result.provider == "azure_openai"
        assert result.tokens_used == 12

    def test_stream(self, provider, monkeypatch):
        lines = [
            'data: {"choices":[{"delta":{"content":"a"}}]}',
            'data: {"choices":[{"delta":{"content":"b"}}]}',
            "data: [DONE]",
        ]
        _patch_client_stream(monkeypatch, lines)
        chunks = list(provider.stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["a", "b"]


# ---------------------------------------------------------------------------
# Bedrock
# ---------------------------------------------------------------------------

class TestBedrockProvider:
    @pytest.fixture()
    def provider(self):
        from llmgate.providers.bedrock import BedrockProvider
        return BedrockProvider({"provider": "bedrock", "model": "anthropic.claude-3-sonnet", "region": "us-east-1"})

    def test_format_request_anthropic(self, provider):
        body = provider._format_request([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ])
        assert body["anthropic_version"] == "bedrock-2023-05-31"
        assert body["system"] == "sys"
        assert all(m["role"] != "system" for m in body["messages"])

    def test_format_request_amazon_titan(self):
        from llmgate.providers.bedrock import BedrockProvider
        p = BedrockProvider({"provider": "bedrock", "model": "amazon.titan-text-express"})
        body = p._format_request([{"role": "user", "content": "hi"}])
        assert "inputText" in body
        assert body["inputText"] == "hi"

    def test_format_request_meta_llama(self):
        from llmgate.providers.bedrock import BedrockProvider
        p = BedrockProvider({"provider": "bedrock", "model": "meta.llama3-8b-instruct"})
        body = p._format_request([{"role": "user", "content": "hi"}])
        assert body["prompt"] == "hi"

    def test_parse_response_anthropic(self, provider):
        data = {"content": [{"text": "hey"}], "usage": {"input_tokens": 5, "output_tokens": 3}}
        text, tokens = provider._parse_response("anthropic.claude-3-sonnet", data)
        assert text == "hey"
        assert tokens == 8

    def test_parse_response_amazon(self, provider):
        data = {"results": [{"outputText": "hello"}], "inputTextTokenCount": 4}
        text, tokens = provider._parse_response("amazon.titan-text-express", data)
        assert text == "hello"
        assert tokens == 4

    def test_parse_response_meta(self, provider):
        data = {"generation": "yo"}
        text, tokens = provider._parse_response("meta.llama3-8b", data)
        assert text == "yo"
        assert tokens is None

    def test_send(self, provider, monkeypatch):
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({
            "content": [{"text": "ok"}],
            "usage": {"input_tokens": 3, "output_tokens": 2},
            "stop_reason": "end_turn",
        }).encode()

        mock_client = MagicMock()
        mock_client.invoke_model.return_value = {"body": mock_body}
        monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

        result = provider.send([{"role": "user", "content": "hi"}])
        assert result.text == "ok"
        assert result.provider == "bedrock"
        assert result.tokens_used == 5

    def test_stream_falls_back_to_send(self, provider, monkeypatch):
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({
            "content": [{"text": "full response"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }).encode()

        mock_client = MagicMock()
        mock_client.invoke_model.return_value = {"body": mock_body}
        monkeypatch.setattr(provider, "_get_client", lambda: mock_client)

        chunks = list(provider.stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["full response"]

    def test_import_error_without_boto3(self, provider, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fail_boto3(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("no boto3")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_boto3)
        with pytest.raises(ImportError, match="boto3"):
            provider._get_client()


# ---------------------------------------------------------------------------
# Vertex AI
# ---------------------------------------------------------------------------

class TestVertexAIProvider:
    @pytest.fixture()
    def provider(self):
        from llmgate.providers.vertexai import VertexAIProvider
        return VertexAIProvider({
            "provider": "vertexai", "model": "gemini-1.5-pro",
            "project_id": "my-project", "region": "us-central1",
        })

    def test_url_construction(self, provider):
        url = provider._get_url()
        assert "us-central1-aiplatform.googleapis.com" in url
        assert "my-project" in url
        assert "gemini-1.5-pro" in url
        assert "generateContent" in url

    def test_url_stream(self, provider):
        url = provider._get_url(stream=True)
        assert "streamGenerateContent" in url

    def test_build_payload_with_system(self, provider):
        payload = provider._build_payload([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ])
        assert payload["systemInstruction"]["parts"][0]["text"] == "sys"
        assert len(payload["contents"]) == 2
        assert payload["contents"][0]["role"] == "user"
        assert payload["contents"][1]["role"] == "model"

    def test_build_payload_gen_config(self, provider):
        payload = provider._build_payload(
            [{"role": "user", "content": "hi"}],
            temperature=0.5, max_tokens=100,
        )
        assert payload["generationConfig"]["temperature"] == 0.5
        assert payload["generationConfig"]["maxOutputTokens"] == 100

    def test_send(self, provider, monkeypatch):
        monkeypatch.setattr(provider, "_get_token", lambda: "fake-token")
        data = {
            "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2},
        }
        _patch_client(monkeypatch, provider, data)
        result = provider.send([{"role": "user", "content": "hello"}])
        assert result.text == "hi"
        assert result.provider == "vertexai"
        assert result.tokens_used == 6

    def test_stream(self, provider, monkeypatch):
        monkeypatch.setattr(provider, "_get_token", lambda: "fake-token")
        lines = [
            'data: {"candidates":[{"content":{"parts":[{"text":"chunk"}]}}]}',
        ]
        _patch_client_stream(monkeypatch, lines)
        chunks = list(provider.stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["chunk"]

    def test_import_error_without_google_auth(self, provider, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fail_google(name, *args, **kwargs):
            if name == "google.auth":
                raise ImportError("no google-auth")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_google)
        with pytest.raises(ImportError, match="google-auth"):
            provider._get_token()


# ---------------------------------------------------------------------------
# HuggingFace
# ---------------------------------------------------------------------------

class TestHuggingFaceProvider:
    def test_chat_model_detection(self):
        from llmgate.providers.huggingface import HuggingFaceProvider
        chat = HuggingFaceProvider({"provider": "huggingface", "model": "mistralai/Mistral-7B-Instruct-v0.3", "api_key": "k"})
        assert chat._is_chat_model() is True

        text = HuggingFaceProvider({"provider": "huggingface", "model": "bigscience/bloom-560m", "api_key": "k"})
        assert text._is_chat_model() is False

    def test_send_chat_model(self, monkeypatch):
        from llmgate.providers.huggingface import HuggingFaceProvider
        p = HuggingFaceProvider({"provider": "huggingface", "model": "meta-llama/Llama-3-8B-Instruct", "api_key": "k"})
        data = {
            "choices": [{"message": {"content": "hey"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 10},
        }
        _patch_client(monkeypatch, p, data)
        result = p.send([{"role": "user", "content": "hi"}])
        assert result.text == "hey"
        assert result.provider == "huggingface"
        assert result.tokens_used == 10

    def test_send_text_generation_model(self, monkeypatch):
        from llmgate.providers.huggingface import HuggingFaceProvider
        p = HuggingFaceProvider({"provider": "huggingface", "model": "bigscience/bloom-560m", "api_key": "k"})
        data = [{"generated_text": "once upon a time"}]
        _patch_client(monkeypatch, p, data)
        result = p.send([{"role": "user", "content": "once"}])
        assert result.text == "once upon a time"
        assert result.tokens_used is None

    def test_headers(self):
        from llmgate.providers.huggingface import HuggingFaceProvider
        p = HuggingFaceProvider({"provider": "huggingface", "model": "m", "api_key": "hf-test"})
        assert p._get_headers()["Authorization"] == "Bearer hf-test"

    def test_stream_falls_back_to_send(self, monkeypatch):
        from llmgate.providers.huggingface import HuggingFaceProvider
        p = HuggingFaceProvider({"provider": "huggingface", "model": "bigscience/bloom-560m", "api_key": "k"})
        data = [{"generated_text": "result"}]
        _patch_client(monkeypatch, p, data)
        chunks = list(p.stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["result"]


# ---------------------------------------------------------------------------
# Replicate
# ---------------------------------------------------------------------------

class TestReplicateProvider:
    @pytest.fixture()
    def provider(self):
        from llmgate.providers.replicate import ReplicateProvider
        return ReplicateProvider({
            "provider": "replicate", "model": "meta/llama-2-70b-chat",
            "api_key": "r8-test", "version": "abc123",
        })

    def test_headers(self, provider):
        headers = provider._get_headers()
        assert headers["Authorization"] == "Bearer r8-test"

    def test_send_polls_until_success(self, provider, monkeypatch):
        create_resp = _make_httpx_response({"id": "pred-123"})
        poll_processing = _make_httpx_response({"status": "processing"})
        poll_done = _make_httpx_response({
            "status": "succeeded",
            "output": ["hello ", "world"],
            "metrics": {"predict_time": 1.5},
        })

        mock_client = MagicMock()
        mock_client.post = MagicMock(return_value=create_resp)
        mock_client.get = MagicMock(side_effect=[poll_processing, poll_done])
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        monkeypatch.setattr(httpx, "Client", MagicMock(return_value=mock_client))

        # Avoid actual sleep
        monkeypatch.setattr("time.sleep", lambda _: None)

        result = provider.send([{"role": "user", "content": "hi"}])
        assert result.text == "hello world"
        assert result.provider == "replicate"
        assert result.finish_reason == "stop"

    def test_send_raises_on_failure(self, provider, monkeypatch):
        create_resp = _make_httpx_response({"id": "pred-456"})
        poll_failed = _make_httpx_response({"status": "failed", "error": "out of memory"})

        mock_client = MagicMock()
        mock_client.post = MagicMock(return_value=create_resp)
        mock_client.get = MagicMock(return_value=poll_failed)
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        monkeypatch.setattr(httpx, "Client", MagicMock(return_value=mock_client))
        monkeypatch.setattr("time.sleep", lambda _: None)

        with pytest.raises(RuntimeError, match="out of memory"):
            provider.send([{"role": "user", "content": "hi"}])

    def test_stream_falls_back_to_send(self, provider, monkeypatch):
        create_resp = _make_httpx_response({"id": "pred-789"})
        poll_done = _make_httpx_response({
            "status": "succeeded", "output": ["done"], "metrics": {},
        })

        mock_client = MagicMock()
        mock_client.post = MagicMock(return_value=create_resp)
        mock_client.get = MagicMock(return_value=poll_done)
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        monkeypatch.setattr(httpx, "Client", MagicMock(return_value=mock_client))
        monkeypatch.setattr("time.sleep", lambda _: None)

        chunks = list(provider.stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["done"]


# ---------------------------------------------------------------------------
# NLP Cloud
# ---------------------------------------------------------------------------

class TestNLPCloudProvider:
    @pytest.fixture()
    def provider(self):
        from llmgate.providers.nlpcloud import NLPCloudProvider
        return NLPCloudProvider({"provider": "nlpcloud", "model": "chatdolphin", "api_key": "nlp-test"})

    def test_headers_use_token(self, provider):
        headers = provider._get_headers()
        assert headers["Authorization"] == "Token nlp-test"

    def test_send(self, provider, monkeypatch):
        data = {"response": "hi there"}
        _patch_client(monkeypatch, provider, data)
        result = provider.send([{"role": "user", "content": "hello"}])
        assert result.text == "hi there"
        assert result.provider == "nlpcloud"
        assert result.tokens_used is None

    def test_history_pairing(self, provider, monkeypatch):
        data = {"response": "ok"}
        client = _patch_client(monkeypatch, provider, data)
        provider.send([
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply1"},
            {"role": "user", "content": "second"},
        ])
        payload = client.post.call_args[1]["json"]
        assert payload["input"] == "second"
        assert len(payload["history"]) == 1
        assert payload["history"][0]["input"] == "first"
        assert payload["history"][0]["response"] == "reply1"

    def test_stream_falls_back_to_send(self, provider, monkeypatch):
        data = {"response": "result"}
        _patch_client(monkeypatch, provider, data)
        chunks = list(provider.stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["result"]
