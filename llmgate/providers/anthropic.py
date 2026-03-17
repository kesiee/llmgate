"""Anthropic Claude provider."""

from __future__ import annotations

import json
from typing import Any, Generator

import httpx

from llmgate.providers.base import BaseProvider
from llmgate.response import LLMResponse


class AnthropicProvider(BaseProvider):
    BASE_URL = "https://api.anthropic.com/v1"

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _get_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.config.get("api_key", ""),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _build_payload(self, messages: list[dict], **kwargs: Any) -> dict:
        # Extract system message if present
        system_text = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                chat_messages.append(msg)

        payload: dict[str, Any] = {
            "model": self.config["model"],
            "messages": chat_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.get("max_tokens", 1024)),
        }
        if system_text:
            payload["system"] = system_text
        for key in ("temperature", "top_p", "stop_sequences"):
            val = kwargs.get(key, self.config.get(key))
            if val is not None:
                payload[key] = val
        return payload

    def send(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        url = f"{self.BASE_URL}/messages"
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=self._get_headers(), json=self._build_payload(messages, **kwargs))
            resp.raise_for_status()
            data = resp.json()
        usage = data.get("usage", {})
        return LLMResponse(
            text=data["content"][0]["text"],
            model=data.get("model", self.config["model"]),
            provider=self.provider_name,
            tokens_used=(usage.get("input_tokens", 0) + usage.get("output_tokens", 0)) or None,
            finish_reason=data.get("stop_reason"),
            raw=data,
        )

    def stream(self, messages: list[dict], **kwargs: Any) -> Generator[str, None, None]:
        url = f"{self.BASE_URL}/messages"
        payload = self._build_payload(messages, **kwargs)
        payload["stream"] = True
        with httpx.Client(timeout=60) as client:
            with client.stream("POST", url, headers=self._get_headers(), json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        if event.get("type") == "content_block_delta":
                            text = event.get("delta", {}).get("text", "")
                            if text:
                                yield text
