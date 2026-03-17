"""Google Gemini provider."""

from __future__ import annotations

import json
from typing import Any, Generator

import httpx

from llmgate.providers.base import BaseProvider
from llmgate.response import LLMResponse


class GeminiProvider(BaseProvider):
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _convert_messages(self, messages: list[dict]) -> tuple[list[dict], str | None]:
        """Convert OpenAI-style messages to Gemini format, extracting system instruction."""
        system_text = None
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        return contents, system_text

    def _build_payload(self, messages: list[dict], **kwargs: Any) -> dict:
        contents, system_text = self._convert_messages(messages)
        payload: dict[str, Any] = {"contents": contents}
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}
        gen_config: dict[str, Any] = {}
        temp = kwargs.get("temperature", self.config.get("temperature"))
        if temp is not None:
            gen_config["temperature"] = temp
        max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens"))
        if max_tokens is not None:
            gen_config["maxOutputTokens"] = max_tokens
        top_p = kwargs.get("top_p", self.config.get("top_p"))
        if top_p is not None:
            gen_config["topP"] = top_p
        if gen_config:
            payload["generationConfig"] = gen_config
        return payload

    def _url(self, action: str = "generateContent") -> str:
        api_key = self.config.get("api_key", "")
        model = self.config["model"]
        return f"{self.BASE_URL}/models/{model}:{action}?key={api_key}"

    def send(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        with httpx.Client(timeout=60) as client:
            resp = client.post(self._url(), json=self._build_payload(messages, **kwargs))
            resp.raise_for_status()
            data = resp.json()
        candidate = data["candidates"][0]
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            text=candidate["content"]["parts"][0]["text"],
            model=self.config["model"],
            provider=self.provider_name,
            tokens_used=(usage.get("promptTokenCount", 0) + usage.get("candidatesTokenCount", 0)) or None,
            finish_reason=candidate.get("finishReason"),
            raw=data,
        )

    def stream(self, messages: list[dict], **kwargs: Any) -> Generator[str, None, None]:
        url = self._url("streamGenerateContent") + "&alt=sse"
        with httpx.Client(timeout=60) as client:
            with client.stream("POST", url, json=self._build_payload(messages, **kwargs)) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        chunk = json.loads(line[6:])
                        candidates = chunk.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts and "text" in parts[0]:
                                yield parts[0]["text"]
