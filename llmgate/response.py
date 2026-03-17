"""LLMResponse dataclass for unified provider responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    tokens_used: int | None
    finish_reason: str | None
    raw: dict[str, Any]

    def __str__(self) -> str:
        return self.text
