"""xAI Grok provider — OpenAI-compatible."""

from llmgate.providers.openai import OpenAIProvider


class XAIProvider(OpenAIProvider):
    BASE_URL = "https://api.x.ai/v1"

    @property
    def provider_name(self) -> str:
        return "xai"
