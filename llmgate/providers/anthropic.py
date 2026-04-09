"""Anthropic Claude provider."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Generator

import httpx

from llmgate.providers.base import BaseProvider
from llmgate.providers.openai import _raise_for_status
from llmgate.response import LLMResponse, ToolCall


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
        # Tool calling
        tools = kwargs.get("tools")
        if tools:
            payload["tools"] = _convert_tools_to_anthropic(tools)
            tool_choice = kwargs.get("tool_choice")
            if tool_choice is not None:
                payload["tool_choice"] = _convert_tool_choice(tool_choice)
        return payload

    def _parse_response(self, data: dict) -> LLMResponse:
        text = ""
        tool_calls = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text = block["text"]
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.get("id", ""),
                    function=block.get("name", ""),
                    arguments=block.get("input", {}),
                ))
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            model=data.get("model", self.config["model"]),
            provider=self.provider_name,
            tokens_used=(usage.get("input_tokens", 0) + usage.get("output_tokens", 0)) or None,
            finish_reason=data.get("stop_reason"),
            raw=data,
            tool_calls=tool_calls,
        )

    def send(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        url = f"{self.BASE_URL}/messages"
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=self._get_headers(), json=self._build_payload(messages, **kwargs))
            _raise_for_status(resp, self.provider_name)
            return self._parse_response(resp.json())

    async def asend(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        url = f"{self.BASE_URL}/messages"
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=self._get_headers(), json=self._build_payload(messages, **kwargs))
            _raise_for_status(resp, self.provider_name)
            return self._parse_response(resp.json())

    def stream(self, messages: list[dict], **kwargs: Any) -> Generator[str, None, None]:
        url = f"{self.BASE_URL}/messages"
        payload = self._build_payload(messages, **kwargs)
        payload["stream"] = True
        with httpx.Client(timeout=60) as client:
            with client.stream("POST", url, headers=self._get_headers(), json=payload) as resp:
                _raise_for_status(resp, self.provider_name)
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        if event.get("type") == "content_block_delta":
                            text = event.get("delta", {}).get("text", "")
                            if text:
                                yield text

    async def astream(self, messages: list[dict], **kwargs: Any) -> AsyncGenerator[str, None]:
        url = f"{self.BASE_URL}/messages"
        payload = self._build_payload(messages, **kwargs)
        payload["stream"] = True
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", url, headers=self._get_headers(), json=payload) as resp:
                _raise_for_status(resp, self.provider_name)
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        if event.get("type") == "content_block_delta":
                            text = event.get("delta", {}).get("text", "")
                            if text:
                                yield text


def _convert_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-style tools to Anthropic format."""
    result = []
    for tool in tools:
        fn = tool.get("function", tool)
        result.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {}),
        })
    return result


def _convert_tool_choice(choice: Any) -> dict:
    if choice == "auto":
        return {"type": "auto"}
    if choice == "none":
        return {"type": "none"}
    if isinstance(choice, str):
        return {"type": "tool", "name": choice}
    return choice
