"""AWS Bedrock provider."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Generator

from llmgate.providers.base import BaseProvider
from llmgate.response import LLMResponse


class BedrockProvider(BaseProvider):

    @property
    def provider_name(self) -> str:
        return "bedrock"

    def _get_client(self) -> Any:
        try:
            import boto3
        except ImportError:
            raise ImportError("AWS Bedrock requires boto3: pip install boto3")
        region = self.config.get("region", "us-east-1")
        return boto3.client("bedrock-runtime", region_name=region)

    def _format_request(self, messages: list[dict], **kwargs: Any) -> dict:
        model_id = self.config["model"]
        max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 1024))
        temp = kwargs.get("temperature", self.config.get("temperature", 0.7))

        # Extract last user message as prompt text
        prompt_text = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                prompt_text = msg["content"]
                break

        if model_id.startswith("anthropic."):
            # Anthropic Claude on Bedrock
            system_text = ""
            chat_msgs = []
            for msg in messages:
                if msg["role"] == "system":
                    system_text = msg["content"]
                else:
                    chat_msgs.append({"role": msg["role"], "content": msg["content"]})
            body: dict[str, Any] = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": chat_msgs,
            }
            if system_text:
                body["system"] = system_text
            if temp is not None:
                body["temperature"] = temp
            return body

        if model_id.startswith("amazon."):
            # Amazon Titan
            return {
                "inputText": prompt_text,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temp,
                },
            }

        if model_id.startswith("meta."):
            # Meta Llama
            return {
                "prompt": prompt_text,
                "max_gen_len": max_tokens,
                "temperature": temp,
            }

        # Default: try Anthropic format
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"],
        }

    def _parse_response(self, model_id: str, data: dict) -> tuple[str, int | None]:
        if model_id.startswith("anthropic."):
            text = data["content"][0]["text"]
            usage = data.get("usage", {})
            tokens = (usage.get("input_tokens", 0) + usage.get("output_tokens", 0)) or None
            return text, tokens

        if model_id.startswith("amazon."):
            results = data.get("results", [{}])
            return results[0].get("outputText", ""), data.get("inputTextTokenCount")

        if model_id.startswith("meta."):
            return data.get("generation", ""), None

        # Fallback
        return data.get("content", [{}])[0].get("text", str(data)), None

    def send(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        client = self._get_client()
        model_id = self.config["model"]
        body = self._format_request(messages, **kwargs)
        response = client.invoke_model(modelId=model_id, body=json.dumps(body))
        data = json.loads(response["body"].read())
        text, tokens = self._parse_response(model_id, data)
        return LLMResponse(
            text=text,
            model=model_id,
            provider=self.provider_name,
            tokens_used=tokens,
            finish_reason=data.get("stop_reason"),
            raw=data,
        )

    async def asend(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        # boto3 does not have native async support; delegate to sync
        return self.send(messages, **kwargs)

    def stream(self, messages: list[dict], **kwargs: Any) -> Generator[str, None, None]:
        # Streaming not trivially supported via invoke_model; return full response
        result = self.send(messages, **kwargs)
        yield result.text

    async def astream(self, messages: list[dict], **kwargs: Any) -> AsyncGenerator[str, None]:
        # boto3 does not have native async support; delegate to sync
        result = await self.asend(messages, **kwargs)
        yield result.text
