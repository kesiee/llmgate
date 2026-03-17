"""Replicate provider — async prediction polling."""

from __future__ import annotations

import time
from typing import Any, Generator

import httpx

from llmgate.providers.base import BaseProvider
from llmgate.response import LLMResponse


class ReplicateProvider(BaseProvider):
    BASE_URL = "https://api.replicate.com/v1"

    @property
    def provider_name(self) -> str:
        return "replicate"

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.get('api_key', '')}",
            "Content-Type": "application/json",
        }

    def send(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        prompt = messages[-1]["content"] if messages else ""
        version = self.config.get("version", "")
        input_data: dict[str, Any] = {"prompt": prompt}
        max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens"))
        if max_tokens:
            input_data["max_tokens"] = max_tokens
        temp = kwargs.get("temperature", self.config.get("temperature"))
        if temp is not None:
            input_data["temperature"] = temp

        headers = self._get_headers()
        with httpx.Client(timeout=60) as client:
            # Create prediction
            resp = client.post(
                f"{self.BASE_URL}/predictions",
                headers=headers,
                json={"version": version, "input": input_data},
            )
            resp.raise_for_status()
            prediction = resp.json()
            prediction_id = prediction["id"]

            # Poll until complete
            for _ in range(120):
                poll = client.get(f"{self.BASE_URL}/predictions/{prediction_id}", headers=headers)
                poll.raise_for_status()
                data = poll.json()
                status = data["status"]
                if status == "succeeded":
                    output = data["output"]
                    text = "".join(output) if isinstance(output, list) else str(output)
                    return LLMResponse(
                        text=text,
                        model=self.config["model"],
                        provider=self.provider_name,
                        tokens_used=data.get("metrics", {}).get("predict_time"),
                        finish_reason="stop",
                        raw=data,
                    )
                if status == "failed":
                    raise RuntimeError(f"Replicate prediction failed: {data.get('error', 'unknown error')}")
                time.sleep(1)

            raise TimeoutError("Replicate prediction timed out after 120 seconds")

    def stream(self, messages: list[dict], **kwargs: Any) -> Generator[str, None, None]:
        # Streaming not supported via polling; return full response
        result = self.send(messages, **kwargs)
        yield result.text
