"""Google Vertex AI provider."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Generator

import httpx

from llmgate.providers.base import BaseProvider
from llmgate.providers.openai import _raise_for_status
from llmgate.response import LLMResponse


class VertexAIProvider(BaseProvider):

    @property
    def provider_name(self) -> str:
        return "vertexai"

    def _get_token(self) -> str:
        try:
            import google.auth
            import google.auth.transport.requests
        except ImportError:
            raise ImportError("Vertex AI requires google-auth: pip install google-auth")
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(google.auth.transport.requests.Request())
        return credentials.token

    def _get_url(self, stream: bool = False) -> str:
        region = self.config.get("region", "us-central1")
        project = self.config["project_id"]
        model = self.config["model"]
        action = "streamGenerateContent" if stream else "generateContent"
        return (
            f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}"
            f"/locations/{region}/publishers/google/models/{model}:{action}"
        )

    def _build_payload(self, messages: list[dict], **kwargs: Any) -> dict:
        contents = []
        system_text = None
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})
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
        if gen_config:
            payload["generationConfig"] = gen_config
        return payload

    def send(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        token = self._get_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        with httpx.Client(timeout=60) as client:
            resp = client.post(self._get_url(), headers=headers, json=self._build_payload(messages, **kwargs))
            _raise_for_status(resp, self.provider_name)
            data = resp.json()
        candidate = data["candidates"][0]
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            text=candidate["content"]["parts"][0]["text"],
            model=self.config["model"],
            provider=self.provider_name,
            tokens_used=(usage.get("promptTokenCount", 0) + usage.get("candidatesTokenCount", 0)) or None,
            finish_reason=candidate.get("finishReason"),
            raw=data,
        )

    async def asend(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        token = self._get_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(self._get_url(), headers=headers, json=self._build_payload(messages, **kwargs))
            _raise_for_status(resp, self.provider_name)
            data = resp.json()
        candidate = data["candidates"][0]
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            text=candidate["content"]["parts"][0]["text"],
            model=self.config["model"],
            provider=self.provider_name,
            tokens_used=(usage.get("promptTokenCount", 0) + usage.get("candidatesTokenCount", 0)) or None,
            finish_reason=candidate.get("finishReason"),
            raw=data,
        )

    def stream(self, messages: list[dict], **kwargs: Any) -> Generator[str, None, None]:
        token = self._get_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = self._get_url(stream=True) + "?alt=sse"
        with httpx.Client(timeout=60) as client:
            with client.stream("POST", url, headers=headers, json=self._build_payload(messages, **kwargs)) as resp:
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
        token = self._get_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = self._get_url(stream=True) + "?alt=sse"
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", url, headers=headers, json=self._build_payload(messages, **kwargs)) as resp:
                _raise_for_status(resp, self.provider_name)
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        chunk = json.loads(line[6:])
                        candidates = chunk.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts and "text" in parts[0]:
                                yield parts[0]["text"]
