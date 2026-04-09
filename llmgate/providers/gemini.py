"""Google Gemini provider."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Generator

import httpx

from llmgate.providers.base import BaseProvider
from llmgate.providers.openai import _raise_for_status
from llmgate.response import EmbeddingResponse, LLMResponse, TokenUsage, ToolCall


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
        # Structured output
        response_format = kwargs.get("response_format")
        if response_format and isinstance(response_format, dict):
            gen_config["responseMimeType"] = "application/json"
        if gen_config:
            payload["generationConfig"] = gen_config
        # Tool calling
        tools = kwargs.get("tools")
        if tools:
            payload["tools"] = [{"functionDeclarations": _convert_tools_to_gemini(tools)}]
        return payload

    def _url(self, action: str = "generateContent") -> str:
        api_key = self.config.get("api_key", "")
        model = self.config["model"]
        return f"{self.BASE_URL}/models/{model}:{action}?key={api_key}"

    def _parse_response(self, data: dict) -> LLMResponse:
        candidate = data["candidates"][0]
        usage = data.get("usageMetadata", {})
        text = ""
        tool_calls = []
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                text = part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(ToolCall(
                    id="",
                    function=fc["name"],
                    arguments=dict(fc.get("args", {})),
                ))
        return LLMResponse(
            text=text,
            model=self.config["model"],
            provider=self.provider_name,
            tokens_used=(usage.get("promptTokenCount", 0) + usage.get("candidatesTokenCount", 0)) or None,
            finish_reason=candidate.get("finishReason"),
            raw=data,
            tool_calls=tool_calls,
        )

    def send(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        with httpx.Client(timeout=60) as client:
            resp = client.post(self._url(), json=self._build_payload(messages, **kwargs))
            _raise_for_status(resp, self.provider_name)
            return self._parse_response(resp.json())

    async def asend(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(self._url(), json=self._build_payload(messages, **kwargs))
            _raise_for_status(resp, self.provider_name)
            return self._parse_response(resp.json())

    def stream(self, messages: list[dict], **kwargs: Any) -> Generator[str, None, None]:
        url = self._url("streamGenerateContent") + "&alt=sse"
        with httpx.Client(timeout=60) as client:
            with client.stream("POST", url, json=self._build_payload(messages, **kwargs)) as resp:
                _raise_for_status(resp, self.provider_name)
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        chunk = json.loads(line[6:])
                        candidates = chunk.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts and "text" in parts[0]:
                                yield parts[0]["text"]

    async def astream(self, messages: list[dict], **kwargs: Any) -> AsyncGenerator[str, None]:
        url = self._url("streamGenerateContent") + "&alt=sse"
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", url, json=self._build_payload(messages, **kwargs)) as resp:
                _raise_for_status(resp, self.provider_name)
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        chunk = json.loads(line[6:])
                        candidates = chunk.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts and "text" in parts[0]:
                                yield parts[0]["text"]

    def embed(self, input: str | list[str], **kwargs: Any) -> EmbeddingResponse:
        if isinstance(input, str):
            input = [input]
        api_key = self.config.get("api_key", "")
        model = self.config["model"]
        url = f"{self.BASE_URL}/models/{model}:batchEmbedContents?key={api_key}"
        requests = [{"model": f"models/{model}", "content": {"parts": [{"text": t}]}} for t in input]
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, json={"requests": requests})
            _raise_for_status(resp, self.provider_name)
            data = resp.json()
        return EmbeddingResponse(
            embeddings=[e["values"] for e in data["embeddings"]],
            model=model,
            provider=self.provider_name,
            usage=TokenUsage(),
            raw=data,
        )

    async def aembed(self, input: str | list[str], **kwargs: Any) -> EmbeddingResponse:
        if isinstance(input, str):
            input = [input]
        api_key = self.config.get("api_key", "")
        model = self.config["model"]
        url = f"{self.BASE_URL}/models/{model}:batchEmbedContents?key={api_key}"
        requests = [{"model": f"models/{model}", "content": {"parts": [{"text": t}]}} for t in input]
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json={"requests": requests})
            _raise_for_status(resp, self.provider_name)
            data = resp.json()
        return EmbeddingResponse(
            embeddings=[e["values"] for e in data["embeddings"]],
            model=model,
            provider=self.provider_name,
            usage=TokenUsage(),
            raw=data,
        )


def _convert_tools_to_gemini(tools: list[dict]) -> list[dict]:
    result = []
    for tool in tools:
        fn = tool.get("function", tool)
        result.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        })
    return result
