"""LM Studio provider — local OpenAI-compatible."""

from llmgate.providers.openai import OpenAIProvider


class LMStudioProvider(OpenAIProvider):
    BASE_URL = "http://localhost:1234/v1"

    @property
    def provider_name(self) -> str:
        return "lmstudio"

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = self.config.get("api_key", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers
