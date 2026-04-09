"""NLP Cloud provider."""

from __future__ import annotations

from typing import Any, AsyncGenerator, Generator

import httpx

from llmgate.providers.base import BaseProvider
from llmgate.providers.openai import _raise_for_status
from llmgate.response import LLMResponse


class NLPCloudProvider(BaseProvider):
    BASE_URL = "https://api.nlpcloud.io/v1"

    @property
    def provider_name(self) -> str:
        return "nlpcloud"

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Token {self.config.get('api_key', '')}",
            "Content-Type": "application/json",
        }

    def _build_chat_body(self, messages: list[dict]) -> dict[str, Any]:
        """Extract current input and history from messages."""
        history: list[dict[str, str]] = []
        current_input = ""
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg["role"] == "user" and i < len(messages) - 1:
                # Pair user/assistant messages as history
                next_msg = messages[i + 1]
                if next_msg["role"] == "assistant":
                    history.append({"input": msg["content"], "response": next_msg["content"]})
                    i += 2
                    continue
            if msg["role"] == "user":
                current_input = msg["content"]
            i += 1

        if not current_input and messages:
            current_input = messages[-1]["content"]

        return {"input": current_input, "history": history}

    def send(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        model = self.config["model"]
        url = f"{self.BASE_URL}/{model}/chatbot"

        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=self._get_headers(), json=self._build_chat_body(messages))
            _raise_for_status(resp, self.provider_name)
            data = resp.json()
        return LLMResponse(
            text=data["response"],
            model=model,
            provider=self.provider_name,
            tokens_used=None,
            finish_reason=None,
            raw=data,
        )

    async def asend(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        model = self.config["model"]
        url = f"{self.BASE_URL}/{model}/chatbot"

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=self._get_headers(), json=self._build_chat_body(messages))
            _raise_for_status(resp, self.provider_name)
            data = resp.json()
        return LLMResponse(
            text=data["response"],
            model=model,
            provider=self.provider_name,
            tokens_used=None,
            finish_reason=None,
            raw=data,
        )

    def stream(self, messages: list[dict], **kwargs: Any) -> Generator[str, None, None]:
        # Streaming not supported; return full response
        result = self.send(messages, **kwargs)
        yield result.text

    async def astream(self, messages: list[dict], **kwargs: Any) -> AsyncGenerator[str, None]:
        # Streaming not supported; return full response
        result = await self.asend(messages, **kwargs)
        yield result.text
