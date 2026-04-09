"""LLMResponse and EmbeddingResponse dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    function: str
    arguments: dict[str, Any]


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    tokens_used: int | None
    finish_reason: str | None
    raw: dict[str, Any]
    tool_calls: list[ToolCall] = field(default_factory=list)
    parsed: Any = None

    def __str__(self) -> str:
        return self.text


@dataclass
class EmbeddingResponse:
    embeddings: list[list[float]]
    model: str
    provider: str
    usage: TokenUsage
    raw: dict[str, Any] = field(default_factory=dict)
