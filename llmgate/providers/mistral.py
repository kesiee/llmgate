"""Mistral AI provider — OpenAI-compatible."""

from llmgate.providers.openai import OpenAIProvider


class MistralProvider(OpenAIProvider):
    BASE_URL = "https://api.mistral.ai/v1"

    @property
    def provider_name(self) -> str:
        return "mistral"
