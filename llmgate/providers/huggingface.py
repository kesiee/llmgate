"""HuggingFace Inference API provider."""

from __future__ import annotations

import re
from typing import Any, AsyncGenerator, Generator

import httpx

from llmgate.providers.base import BaseProvider
from llmgate.providers.openai import _raise_for_status
from llmgate.response import LLMResponse

_CHAT_PATTERN = re.compile(r"(instruct|chat|\bit\b)", re.IGNORECASE)


class HuggingFaceProvider(BaseProvider):
    BASE_URL = "https://api-inference.huggingface.co"

    @property
    def provider_name(self) -> str:
        return "huggingface"

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.get('api_key', '')}",
            "Content-Type": "application/json",
        }

    def _is_chat_model(self) -> bool:
        return bool(_CHAT_PATTERN.search(self.config["model"]))

    def send(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        model = self.config["model"]
        headers = self._get_headers()

        if self._is_chat_model():
            url = f"{self.BASE_URL}/models/{model}/v1/chat/completions"
            payload: dict[str, Any] = {"model": model, "messages": messages}
            max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens"))
            if max_tokens:
                payload["max_tokens"] = max_tokens
            temp = kwargs.get("temperature", self.config.get("temperature"))
            if temp is not None:
                payload["temperature"] = temp
            with httpx.Client(timeout=60) as client:
                resp = client.post(url, headers=headers, json=payload)
                _raise_for_status(resp, self.provider_name)
                data = resp.json()
            return LLMResponse(
                text=data["choices"][0]["message"]["content"],
                model=model,
                provider=self.provider_name,
                tokens_used=data.get("usage", {}).get("total_tokens"),
                finish_reason=data["choices"][0].get("finish_reason"),
                raw=data,
            )
        else:
            # Raw text generation
            prompt = messages[-1]["content"] if messages else ""
            url = f"{self.BASE_URL}/models/{model}"
            payload = {"inputs": prompt}
            params: dict[str, Any] = {}
            max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens"))
            if max_tokens:
                params["max_new_tokens"] = max_tokens
            temp = kwargs.get("temperature", self.config.get("temperature"))
            if temp is not None:
                params["temperature"] = temp
            if params:
                payload["parameters"] = params
            with httpx.Client(timeout=60) as client:
                resp = client.post(url, headers=headers, json=payload)
                _raise_for_status(resp, self.provider_name)
                data = resp.json()
            text = data[0]["generated_text"] if isinstance(data, list) else data.get("generated_text", str(data))
            return LLMResponse(
                text=text,
                model=model,
                provider=self.provider_name,
                tokens_used=None,
                finish_reason=None,
                raw=data if isinstance(data, dict) else {"output": data},
            )

    async def asend(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        model = self.config["model"]
        headers = self._get_headers()

        if self._is_chat_model():
            url = f"{self.BASE_URL}/models/{model}/v1/chat/completions"
            payload: dict[str, Any] = {"model": model, "messages": messages}
            max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens"))
            if max_tokens:
                payload["max_tokens"] = max_tokens
            temp = kwargs.get("temperature", self.config.get("temperature"))
            if temp is not None:
                payload["temperature"] = temp
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(url, headers=headers, json=payload)
                _raise_for_status(resp, self.provider_name)
                data = resp.json()
            return LLMResponse(
                text=data["choices"][0]["message"]["content"],
                model=model,
                provider=self.provider_name,
                tokens_used=data.get("usage", {}).get("total_tokens"),
                finish_reason=data["choices"][0].get("finish_reason"),
                raw=data,
            )
        else:
            # Raw text generation
            prompt = messages[-1]["content"] if messages else ""
            url = f"{self.BASE_URL}/models/{model}"
            payload = {"inputs": prompt}
            params: dict[str, Any] = {}
            max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens"))
            if max_tokens:
                params["max_new_tokens"] = max_tokens
            temp = kwargs.get("temperature", self.config.get("temperature"))
            if temp is not None:
                params["temperature"] = temp
            if params:
                payload["parameters"] = params
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(url, headers=headers, json=payload)
                _raise_for_status(resp, self.provider_name)
                data = resp.json()
            text = data[0]["generated_text"] if isinstance(data, list) else data.get("generated_text", str(data))
            return LLMResponse(
                text=text,
                model=model,
                provider=self.provider_name,
                tokens_used=None,
                finish_reason=None,
                raw=data if isinstance(data, dict) else {"output": data},
            )

    def stream(self, messages: list[dict], **kwargs: Any) -> Generator[str, None, None]:
        # Streaming varies by model; return full response
        result = self.send(messages, **kwargs)
        yield result.text

    async def astream(self, messages: list[dict], **kwargs: Any) -> AsyncGenerator[str, None]:
        # Streaming varies by model; return full response
        result = await self.asend(messages, **kwargs)
        yield result.text
