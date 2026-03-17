"""Groq provider — OpenAI-compatible."""

from llmgate.providers.openai import OpenAIProvider


class GroqProvider(OpenAIProvider):
    BASE_URL = "https://api.groq.com/openai/v1"

    @property
    def provider_name(self) -> str:
        return "groq"
