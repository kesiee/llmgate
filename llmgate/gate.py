"""LLMGate — unified gateway for multiple LLM providers."""

from __future__ import annotations

import importlib
from typing import Any, Generator

from llmgate.config import load_config
from llmgate.providers.base import BaseProvider
from llmgate.response import LLMResponse

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


class LLMGate:
    """Unified gateway for calling any LLM provider."""

    def __init__(
        self,
        config_path: str | None = None,
        profile: str | None = None,
    ) -> None:
        self._config_path = config_path
        self._profile = profile
        self._config = load_config(config_path, profile)
        self._provider = self._init_provider()

    def _init_provider(self) -> BaseProvider:
        provider_name = self._config["provider"]
        cls = _load_provider_class(provider_name)
        return cls(self._config)

    def switch(self, profile: str) -> None:
        """Hot-swap to a different profile."""
        self._profile = profile
        self._config = load_config(self._config_path, profile)
        self._provider = self._init_provider()

    def chat(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Send a chat message and return the response."""
        messages = [{"role": "user", "content": prompt}]
        return self._provider.send(messages, **kwargs)

    def chat_messages(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        """Send a full message list and return the response."""
        return self._provider.send(messages, **kwargs)

    def stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        """Stream a chat response, yielding text chunks."""
        messages = [{"role": "user", "content": prompt}]
        yield from self._provider.stream(messages, **kwargs)

    def stream_messages(self, messages: list[dict], **kwargs: Any) -> Generator[str, None, None]:
        """Stream a response from a full message list."""
        yield from self._provider.stream(messages, **kwargs)

    @property
    def provider_name(self) -> str:
        return self._config["provider"]

    @property
    def model(self) -> str:
        return self._config["model"]

    @property
    def config(self) -> dict[str, Any]:
        return dict(self._config)
