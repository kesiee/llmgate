"""Custom exception hierarchy for llmgate."""

from __future__ import annotations


class LLMGateError(Exception):
    """Base exception for all llmgate errors."""


class AuthError(LLMGateError):
    """Raised on 401/403 — bad or missing API key."""

    def __init__(self, provider: str, message: str = "") -> None:
        self.provider = provider
        super().__init__(message or f"Authentication failed for {provider}")


class RateLimitError(LLMGateError):
    """Raised on 429 — rate or quota exceeded."""

    def __init__(self, provider: str, message: str = "") -> None:
        self.provider = provider
        super().__init__(message or f"Rate limited by {provider}")


class ProviderAPIError(LLMGateError):
    """Raised on other provider HTTP errors."""

    def __init__(self, provider: str, status_code: int, message: str = "") -> None:
        self.provider = provider
        self.status_code = status_code
        super().__init__(message or f"{provider} returned HTTP {status_code}")


class ModelNotFoundError(LLMGateError):
    """Raised when a model or provider can't be resolved."""

    def __init__(self, model: str, message: str = "") -> None:
        self.model = model
        super().__init__(message or f"Unknown model: {model}")


class EmbeddingsNotSupported(LLMGateError):
    """Raised when a provider doesn't offer an embeddings API."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(f"{provider} does not support embeddings")
