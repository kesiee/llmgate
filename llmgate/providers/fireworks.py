"""Fireworks AI provider — OpenAI-compatible."""

from llmgate.providers.openai import OpenAIProvider


class FireworksProvider(OpenAIProvider):
    BASE_URL = "https://api.fireworks.ai/inference/v1"

    @property
    def provider_name(self) -> str:
        return "fireworks"
