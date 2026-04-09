"""Azure OpenAI provider."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Generator

import httpx

from llmgate.providers.base import BaseProvider
from llmgate.providers.openai import _raise_for_status
from llmgate.response import LLMResponse


class AzureOpenAIProvider(BaseProvider):

    @property
    def provider_name(self) -> str:
        return "azure_openai"

    def _get_headers(self) -> dict[str, str]:
        return {
            "api-key": self.config.get("api_key", ""),
            "Content-Type": "application/json",
        }

    def _get_url(self) -> str:
        resource = self.config["resource_name"]
        deployment = self.config["deployment_name"]
        api_version = self.config.get("api_version", "2024-02-01")
        return (
            f"https://{resource}.openai.azure.com/openai/deployments/{deployment}"
            f"/chat/completions?api-version={api_version}"
        )

    def _build_payload(self, messages: list[dict], **kwargs: Any) -> dict:
        payload: dict[str, Any] = {"messages": messages}
        for key in ("temperature", "max_tokens", "top_p", "stop"):
            val = kwargs.get(key, self.config.get(key))
            if val is not None:
                payload[key] = val
        return payload

    def send(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        with httpx.Client(timeout=60) as client:
            resp = client.post(self._get_url(), headers=self._get_headers(), json=self._build_payload(messages, **kwargs))
            _raise_for_status(resp, self.provider_name)
            data = resp.json()
        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=data.get("model", self.config.get("model", "")),
            provider=self.provider_name,
            tokens_used=data.get("usage", {}).get("total_tokens"),
            finish_reason=data["choices"][0].get("finish_reason"),
            raw=data,
        )

    async def asend(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(self._get_url(), headers=self._get_headers(), json=self._build_payload(messages, **kwargs))
            _raise_for_status(resp, self.provider_name)
            data = resp.json()
        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=data.get("model", self.config.get("model", "")),
            provider=self.provider_name,
            tokens_used=data.get("usage", {}).get("total_tokens"),
            finish_reason=data["choices"][0].get("finish_reason"),
            raw=data,
        )

    def stream(self, messages: list[dict], **kwargs: Any) -> Generator[str, None, None]:
        payload = self._build_payload(messages, **kwargs)
        payload["stream"] = True
        with httpx.Client(timeout=60) as client:
            with client.stream("POST", self._get_url(), headers=self._get_headers(), json=payload) as resp:
                _raise_for_status(resp, self.provider_name)
                for line in resp.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content

    async def astream(self, messages: list[dict], **kwargs: Any) -> AsyncGenerator[str, None]:
        payload = self._build_payload(messages, **kwargs)
        payload["stream"] = True
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", self._get_url(), headers=self._get_headers(), json=payload) as resp:
                _raise_for_status(resp, self.provider_name)
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
