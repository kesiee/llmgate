"""Cohere provider."""

from __future__ import annotations

import json
from typing import Any, Generator

import httpx

from llmgate.providers.base import BaseProvider
from llmgate.response import LLMResponse


class CohereProvider(BaseProvider):
    BASE_URL = "https://api.cohere.com/v2"

    @property
    def provider_name(self) -> str:
        return "cohere"

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.get('api_key', '')}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, messages: list[dict], **kwargs: Any) -> dict:
        payload: dict[str, Any] = {
            "model": self.config["model"],
            "messages": messages,
        }
        for key in ("temperature", "max_tokens"):
            val = kwargs.get(key, self.config.get(key))
            if val is not None:
                payload[key] = val
        return payload

    def send(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        url = f"{self.BASE_URL}/chat"
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=self._get_headers(), json=self._build_payload(messages, **kwargs))
            resp.raise_for_status()
            data = resp.json()
        usage = data.get("usage", {}).get("tokens", {})
        return LLMResponse(
            text=data["message"]["content"][0]["text"],
            model=data.get("model", self.config["model"]),
            provider=self.provider_name,
            tokens_used=(usage.get("input_tokens", 0) + usage.get("output_tokens", 0)) or None,
            finish_reason=data.get("finish_reason"),
            raw=data,
        )

    def stream(self, messages: list[dict], **kwargs: Any) -> Generator[str, None, None]:
        url = f"{self.BASE_URL}/chat"
        payload = self._build_payload(messages, **kwargs)
        payload["stream"] = True
        with httpx.Client(timeout=60) as client:
            with client.stream("POST", url, headers=self._get_headers(), json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        if event.get("type") == "content-delta":
                            text = event.get("delta", {}).get("message", {}).get("content", {}).get("text", "")
                            if text:
                                yield text
