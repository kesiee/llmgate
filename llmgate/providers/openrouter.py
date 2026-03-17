"""OpenRouter provider — OpenAI-compatible with extra headers."""

from llmgate.providers.openai import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    BASE_URL = "https://openrouter.ai/api/v1"

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def _get_headers(self) -> dict[str, str]:
        headers = super()._get_headers()
        site_url = self.config.get("site_url", "")
        if site_url:
            headers["HTTP-Referer"] = site_url
        app_name = self.config.get("app_name", "")
        if app_name:
            headers["X-Title"] = app_name
        return headers
