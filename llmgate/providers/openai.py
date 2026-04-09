"""OpenAI provider — base for all OpenAI-compatible APIs."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Generator

import httpx

from llmgate.exceptions import AuthError, ProviderAPIError, RateLimitError
from llmgate.providers.base import BaseProvider
from llmgate.response import EmbeddingResponse, LLMResponse, TokenUsage, ToolCall


def _raise_for_status(resp: httpx.Response, provider: str) -> None:
    """Convert HTTP errors to llmgate exceptions."""
    if resp.is_success:
        return
    if resp.status_code in (401, 403):
        raise AuthError(provider, resp.text)
    if resp.status_code == 429:
        raise RateLimitError(provider, resp.text)
    raise ProviderAPIError(provider, resp.status_code, resp.text)


def _extract_tool_calls(choice: dict) -> list[ToolCall]:
    """Extract tool calls from a response choice."""
    raw_calls = choice.get("message", {}).get("tool_calls", [])
    result = []
    for tc in raw_calls:
        fn = tc.get("function", {})
        args = fn.get("arguments", "{}")
        result.append(ToolCall(
            id=tc.get("id", ""),
            function=fn.get("name", ""),
            arguments=json.loads(args) if isinstance(args, str) else args,
        ))
    return result


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

    def _get_embed_url(self) -> str:
        base = self.config.get("base_url", self.BASE_URL).rstrip("/")
        return f"{base}/embeddings"

    def _build_payload(self, messages: list[dict], **kwargs: Any) -> dict:
        payload: dict[str, Any] = {
            "model": self.config["model"],
            "messages": messages,
        }
        for key in ("temperature", "max_tokens", "top_p", "stop", "frequency_penalty", "presence_penalty"):
            val = kwargs.get(key, self.config.get(key))
            if val is not None:
                payload[key] = val
        # Tool calling
        tools = kwargs.get("tools")
        if tools:
            payload["tools"] = tools
            tool_choice = kwargs.get("tool_choice")
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        # JSON mode / response format
        response_format = kwargs.get("response_format")
        if response_format and isinstance(response_format, dict):
            payload["response_format"] = response_format
        return payload

    def _parse_response(self, data: dict) -> LLMResponse:
        choice = data["choices"][0]
        return LLMResponse(
            text=choice["message"].get("content") or "",
            model=data.get("model", self.config["model"]),
            provider=self.provider_name,
            tokens_used=data.get("usage", {}).get("total_tokens"),
            finish_reason=choice.get("finish_reason"),
            raw=data,
            tool_calls=_extract_tool_calls(choice),
        )

    def send(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                self._get_url(),
                headers=self._get_headers(),
                json=self._build_payload(messages, **kwargs),
            )
            _raise_for_status(resp, self.provider_name)
            return self._parse_response(resp.json())

    async def asend(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self._get_url(),
                headers=self._get_headers(),
                json=self._build_payload(messages, **kwargs),
            )
            _raise_for_status(resp, self.provider_name)
            return self._parse_response(resp.json())

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
            async with client.stream(
                "POST",
                self._get_url(),
                headers=self._get_headers(),
                json=payload,
            ) as resp:
                _raise_for_status(resp, self.provider_name)
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content

    def embed(self, input: str | list[str], **kwargs: Any) -> EmbeddingResponse:
        if isinstance(input, str):
            input = [input]
        payload: dict[str, Any] = {"model": self.config["model"], "input": input}
        dimensions = kwargs.get("dimensions")
        if dimensions:
            payload["dimensions"] = dimensions
        with httpx.Client(timeout=60) as client:
            resp = client.post(self._get_embed_url(), headers=self._get_headers(), json=payload)
            _raise_for_status(resp, self.provider_name)
            data = resp.json()
        return EmbeddingResponse(
            embeddings=[d["embedding"] for d in data["data"]],
            model=data.get("model", self.config["model"]),
            provider=self.provider_name,
            usage=TokenUsage(
                prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                total_tokens=data.get("usage", {}).get("total_tokens", 0),
            ),
            raw=data,
        )

    async def aembed(self, input: str | list[str], **kwargs: Any) -> EmbeddingResponse:
        if isinstance(input, str):
            input = [input]
        payload: dict[str, Any] = {"model": self.config["model"], "input": input}
        dimensions = kwargs.get("dimensions")
        if dimensions:
            payload["dimensions"] = dimensions
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(self._get_embed_url(), headers=self._get_headers(), json=payload)
            _raise_for_status(resp, self.provider_name)
            data = resp.json()
        return EmbeddingResponse(
            embeddings=[d["embedding"] for d in data["data"]],
            model=data.get("model", self.config["model"]),
            provider=self.provider_name,
            usage=TokenUsage(
                prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                total_tokens=data.get("usage", {}).get("total_tokens", 0),
            ),
            raw=data,
        )
