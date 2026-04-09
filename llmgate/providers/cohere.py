"""Cohere provider."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Generator

import httpx

from llmgate.providers.base import BaseProvider
from llmgate.providers.openai import _raise_for_status
from llmgate.response import EmbeddingResponse, LLMResponse, TokenUsage


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
            _raise_for_status(resp, self.provider_name)
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

    async def asend(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        url = f"{self.BASE_URL}/chat"
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=self._get_headers(), json=self._build_payload(messages, **kwargs))
            _raise_for_status(resp, self.provider_name)
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
                _raise_for_status(resp, self.provider_name)
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        if event.get("type") == "content-delta":
                            text = event.get("delta", {}).get("message", {}).get("content", {}).get("text", "")
                            if text:
                                yield text

    async def astream(self, messages: list[dict], **kwargs: Any) -> AsyncGenerator[str, None]:
        url = f"{self.BASE_URL}/chat"
        payload = self._build_payload(messages, **kwargs)
        payload["stream"] = True
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", url, headers=self._get_headers(), json=payload) as resp:
                _raise_for_status(resp, self.provider_name)
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        if event.get("type") == "content-delta":
                            text = event.get("delta", {}).get("message", {}).get("content", {}).get("text", "")
                            if text:
                                yield text

    def embed(self, input: str | list[str], **kwargs: Any) -> EmbeddingResponse:
        if isinstance(input, str):
            input = [input]
        url = f"{self.BASE_URL}/embed"
        payload: dict[str, Any] = {
            "texts": input,
            "model": self.config["model"],
            "input_type": "search_document",
        }
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=self._get_headers(), json=payload)
            _raise_for_status(resp, self.provider_name)
            data = resp.json()
        meta = data.get("meta", {}).get("billed_units", {})
        return EmbeddingResponse(
            embeddings=data["embeddings"].get("float", data["embeddings"]) if isinstance(data["embeddings"], dict) else data["embeddings"],
            model=data.get("model", self.config["model"]),
            provider=self.provider_name,
            usage=TokenUsage(
                prompt_tokens=meta.get("input_tokens", 0),
                total_tokens=meta.get("input_tokens", 0),
            ),
            raw=data,
        )

    async def aembed(self, input: str | list[str], **kwargs: Any) -> EmbeddingResponse:
        if isinstance(input, str):
            input = [input]
        url = f"{self.BASE_URL}/embed"
        payload: dict[str, Any] = {
            "texts": input,
            "model": self.config["model"],
            "input_type": "search_document",
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=self._get_headers(), json=payload)
            _raise_for_status(resp, self.provider_name)
            data = resp.json()
        meta = data.get("meta", {}).get("billed_units", {})
        return EmbeddingResponse(
            embeddings=data["embeddings"].get("float", data["embeddings"]) if isinstance(data["embeddings"], dict) else data["embeddings"],
            model=data.get("model", self.config["model"]),
            provider=self.provider_name,
            usage=TokenUsage(
                prompt_tokens=meta.get("input_tokens", 0),
                total_tokens=meta.get("input_tokens", 0),
            ),
            raw=data,
        )
