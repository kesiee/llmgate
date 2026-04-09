"""LLMGate — unified gateway for multiple LLM providers."""

from __future__ import annotations

import importlib
import json
from typing import Any, AsyncGenerator, Generator

from llmgate.config import load_config
from llmgate.middleware.base import BaseMiddleware
from llmgate.providers.base import BaseProvider
from llmgate.response import EmbeddingResponse, LLMResponse

PROVIDER_REGISTRY: dict[str, str] = {
    "openai": "llmgate.providers.openai:OpenAIProvider",
    "anthropic": "llmgate.providers.anthropic:AnthropicProvider",
    "gemini": "llmgate.providers.gemini:GeminiProvider",
    "cohere": "llmgate.providers.cohere:CohereProvider",
    "ollama": "llmgate.providers.ollama:OllamaProvider",
    "groq": "llmgate.providers.groq:GroqProvider",
    "mistral": "llmgate.providers.mistral:MistralProvider",
    "openrouter": "llmgate.providers.openrouter:OpenRouterProvider",
    "together": "llmgate.providers.together:TogetherProvider",
    "fireworks": "llmgate.providers.fireworks:FireworksProvider",
    "perplexity": "llmgate.providers.perplexity:PerplexityProvider",
    "deepseek": "llmgate.providers.deepseek:DeepSeekProvider",
    "xai": "llmgate.providers.xai:XAIProvider",
    "azure_openai": "llmgate.providers.azure_openai:AzureOpenAIProvider",
    "bedrock": "llmgate.providers.bedrock:BedrockProvider",
    "vertexai": "llmgate.providers.vertexai:VertexAIProvider",
    "huggingface": "llmgate.providers.huggingface:HuggingFaceProvider",
    "replicate": "llmgate.providers.replicate:ReplicateProvider",
    "ai21": "llmgate.providers.ai21:AI21Provider",
    "nlpcloud": "llmgate.providers.nlpcloud:NLPCloudProvider",
    "lmstudio": "llmgate.providers.lmstudio:LMStudioProvider",
}


def _load_provider_class(provider_name: str) -> type[BaseProvider]:
    """Lazily import a provider class from the registry."""
    if provider_name not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Available: {sorted(PROVIDER_REGISTRY.keys())}"
        )
    module_path, class_name = PROVIDER_REGISTRY[provider_name].rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _apply_structured_output(response: LLMResponse, response_format: Any) -> LLMResponse:
    """Parse response text into a Pydantic model if response_format is a class."""
    if response_format is None:
        return response
    # If it's a dict (e.g. {"type": "json_object"}), skip parsing
    if isinstance(response_format, dict):
        return response
    # Assume it's a Pydantic model class
    try:
        text = response.text.strip()
        # Try to extract JSON from markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                if line.startswith("```") and in_block:
                    break
                if in_block:
                    json_lines.append(line)
            text = "\n".join(json_lines)
        data = json.loads(text)
        response.parsed = response_format(**data)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return response


class LLMGate:
    """Unified gateway for calling any LLM provider."""

    def __init__(
        self,
        config_path: str | None = None,
        profile: str | None = None,
        middleware: list[BaseMiddleware] | None = None,
    ) -> None:
        self._config_path = config_path
        self._profile = profile
        self._config = load_config(config_path, profile)
        self._provider = self._init_provider()
        self._middleware: list[BaseMiddleware] = middleware or []

    def _init_provider(self) -> BaseProvider:
        provider_name = self._config["provider"]
        cls = _load_provider_class(provider_name)
        return cls(self._config)

    def switch(self, profile: str) -> None:
        """Hot-swap to a different profile."""
        self._profile = profile
        self._config = load_config(self._config_path, profile)
        self._provider = self._init_provider()

    # ------------------------------------------------------------------
    # Sync chat
    # ------------------------------------------------------------------

    def chat(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Send a chat message and return the response."""
        messages = [{"role": "user", "content": prompt}]
        return self._send_with_middleware(messages, **kwargs)

    def chat_messages(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        """Send a full message list and return the response."""
        return self._send_with_middleware(messages, **kwargs)

    def _send_with_middleware(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        response_format = kwargs.get("response_format")

        def call_next(msgs: list[dict], **kw: Any) -> LLMResponse:
            # Inject JSON mode for structured output
            if response_format and not isinstance(response_format, dict):
                kw["response_format"] = {"type": "json_object"}
            return self._provider.send(msgs, **kw)

        chain = call_next
        for mw in reversed(self._middleware):
            prev = chain
            chain = lambda msgs, _mw=mw, _next=prev, **kw: _mw.handle(msgs, _next, **kw)

        result = chain(messages, **kwargs)
        return _apply_structured_output(result, response_format)

    # ------------------------------------------------------------------
    # Async chat
    # ------------------------------------------------------------------

    async def achat(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Async version of chat()."""
        messages = [{"role": "user", "content": prompt}]
        return await self._asend_with_middleware(messages, **kwargs)

    async def achat_messages(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        """Async version of chat_messages()."""
        return await self._asend_with_middleware(messages, **kwargs)

    async def _asend_with_middleware(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        response_format = kwargs.get("response_format")

        async def call_next(msgs: list[dict], **kw: Any) -> LLMResponse:
            if response_format and not isinstance(response_format, dict):
                kw["response_format"] = {"type": "json_object"}
            return await self._provider.asend(msgs, **kw)

        chain = call_next
        for mw in reversed(self._middleware):
            prev = chain
            chain = lambda msgs, _mw=mw, _next=prev, **kw: _mw.ahandle(msgs, _next, **kw)

        result = await chain(messages, **kwargs)
        return _apply_structured_output(result, response_format)

    # ------------------------------------------------------------------
    # Sync streaming
    # ------------------------------------------------------------------

    def stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        """Stream a chat response, yielding text chunks."""
        messages = [{"role": "user", "content": prompt}]
        yield from self._provider.stream(messages, **kwargs)

    def stream_messages(self, messages: list[dict], **kwargs: Any) -> Generator[str, None, None]:
        """Stream a response from a full message list."""
        yield from self._provider.stream(messages, **kwargs)

    # ------------------------------------------------------------------
    # Async streaming
    # ------------------------------------------------------------------

    async def astream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Async stream a chat response, yielding text chunks."""
        messages = [{"role": "user", "content": prompt}]
        async for chunk in self._provider.astream(messages, **kwargs):
            yield chunk

    async def astream_messages(self, messages: list[dict], **kwargs: Any) -> AsyncGenerator[str, None]:
        """Async stream a response from a full message list."""
        async for chunk in self._provider.astream(messages, **kwargs):
            yield chunk

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, input: str | list[str], **kwargs: Any) -> EmbeddingResponse:
        """Generate embeddings using the current provider."""
        return self._provider.embed(input, **kwargs)

    async def aembed(self, input: str | list[str], **kwargs: Any) -> EmbeddingResponse:
        """Async version of embed()."""
        return await self._provider.aembed(input, **kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return self._config["provider"]

    @property
    def model(self) -> str:
        return self._config["model"]

    @property
    def config(self) -> dict[str, Any]:
        return dict(self._config)
