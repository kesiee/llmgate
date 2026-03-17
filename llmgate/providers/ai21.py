"""AI21 provider — OpenAI-compatible for Jamba models."""

from llmgate.providers.openai import OpenAIProvider


class AI21Provider(OpenAIProvider):
    BASE_URL = "https://api.ai21.com/studio/v1"

    @property
    def provider_name(self) -> str:
        return "ai21"
