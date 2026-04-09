"""Ollama provider — local LLM server."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Generator

import httpx

from llmgate.providers.base import BaseProvider
from llmgate.providers.openai import _raise_for_status
from llmgate.response import LLMResponse


class OllamaProvider(BaseProvider):
    BASE_URL = "http://localhost:11434"

    @property
    def provider_name(self) -> str:
        return "ollama"

    def _get_url(self) -> str:
        base = self.config.get("base_url", self.BASE_URL).rstrip("/")
        return f"{base}/api/chat"

    def _build_payload(self, messages: list[dict], **kwargs: Any) -> dict:
        payload: dict[str, Any] = {
            "model": self.config["model"],
            "messages": messages,
        }
        options: dict[str, Any] = {}
        temp = kwargs.get("temperature", self.config.get("temperature"))
        if temp is not None:
            options["temperature"] = temp
        if options:
            payload["options"] = options
        return payload

    def send(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        payload = self._build_payload(messages, **kwargs)
        payload["stream"] = False
        with httpx.Client(timeout=60) as client:
            resp = client.post(self._get_url(), json=payload)
            _raise_for_status(resp, self.provider_name)
            data = resp.json()
        return LLMResponse(
            text=data["message"]["content"],
            model=data.get("model", self.config["model"]),
            provider=self.provider_name,
            tokens_used=data.get("eval_count"),
            finish_reason=data.get("done_reason"),
            raw=data,
        )

    async def asend(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        payload = self._build_payload(messages, **kwargs)
        payload["stream"] = False
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(self._get_url(), json=payload)
            _raise_for_status(resp, self.provider_name)
            data = resp.json()
        return LLMResponse(
            text=data["message"]["content"],
            model=data.get("model", self.config["model"]),
            provider=self.provider_name,
            tokens_used=data.get("eval_count"),
            finish_reason=data.get("done_reason"),
            raw=data,
        )

    def stream(self, messages: list[dict], **kwargs: Any) -> Generator[str, None, None]:
        payload = self._build_payload(messages, **kwargs)
        payload["stream"] = True
        with httpx.Client(timeout=60) as client:
            with client.stream("POST", self._get_url(), json=payload) as resp:
                _raise_for_status(resp, self.provider_name)
                for line in resp.iter_lines():
                    if line.strip():
                        chunk = json.loads(line)
                        text = chunk.get("message", {}).get("content", "")
                        if text:
                            yield text

    async def astream(self, messages: list[dict], **kwargs: Any) -> AsyncGenerator[str, None]:
        payload = self._build_payload(messages, **kwargs)
        payload["stream"] = True
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", self._get_url(), json=payload) as resp:
                _raise_for_status(resp, self.provider_name)
                async for line in resp.aiter_lines():
                    if line.strip():
                        chunk = json.loads(line)
                        text = chunk.get("message", {}).get("content", "")
                        if text:
                            yield text
