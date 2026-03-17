"""OpenAI provider — base for all OpenAI-compatible APIs."""

from __future__ import annotations

import json
from typing import Any, Generator

import httpx

from llmgate.providers.base import BaseProvider
from llmgate.response import LLMResponse


class OpenAIProvider(BaseProvider):
    BASE_URL = "https://api.openai.com/v1"

    @property
    def provider_name(self) -> str:
        return "openai"

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = self.config.get("api_key", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _get_url(self) -> str:
        base = self.config.get("base_url", self.BASE_URL).rstrip("/")
        return f"{base}/chat/completions"

    def _build_payload(self, messages: list[dict], **kwargs: Any) -> dict:
        payload: dict[str, Any] = {
            "model": self.config["model"],
            "messages": messages,
        }
        for key in ("temperature", "max_tokens", "top_p", "stop", "frequency_penalty", "presence_penalty"):
            val = kwargs.get(key, self.config.get(key))
            if val is not None:
                payload[key] = val
        return payload

    def send(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                self._get_url(),
                headers=self._get_headers(),
                json=self._build_payload(messages, **kwargs),
            )
            resp.raise_for_status()
            data = resp.json()
        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=data.get("model", self.config["model"]),
            provider=self.provider_name,
            tokens_used=data.get("usage", {}).get("total_tokens"),
            finish_reason=data["choices"][0].get("finish_reason"),
            raw=data,
        )

    def stream(self, messages: list[dict], **kwargs: Any) -> Generator[str, None, None]:
        payload = self._build_payload(messages, **kwargs)
        payload["stream"] = True
        with httpx.Client(timeout=60) as client:
            with client.stream(
                "POST",
                self._get_url(),
                headers=self._get_headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
