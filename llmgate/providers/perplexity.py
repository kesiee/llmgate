"""Perplexity AI provider — OpenAI-compatible."""

from llmgate.providers.openai import OpenAIProvider


class PerplexityProvider(OpenAIProvider):
    BASE_URL = "https://api.perplexity.ai"

    @property
    def provider_name(self) -> str:
        return "perplexity"
