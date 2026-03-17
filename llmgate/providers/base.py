"""Abstract base provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generator


class BaseProvider(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def send(self, messages: list[dict], **kwargs: Any) -> Any:
        ...

    @abstractmethod
    def stream(self, messages: list[dict], **kwargs: Any) -> Generator[str, None, None]:
        ...
