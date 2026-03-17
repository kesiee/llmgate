"""Together AI provider — OpenAI-compatible."""

from llmgate.providers.openai import OpenAIProvider


class TogetherProvider(OpenAIProvider):
    BASE_URL = "https://api.together.xyz/v1"

    @property
    def provider_name(self) -> str:
        return "together"
