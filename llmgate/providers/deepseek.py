"""DeepSeek provider — OpenAI-compatible."""

from llmgate.providers.openai import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    BASE_URL = "https://api.deepseek.com/v1"

    @property
    def provider_name(self) -> str:
        return "deepseek"
