"""llmgate — plug-and-play LLM connector via YAML config."""

from llmgate.gate import LLMGate
from llmgate.response import EmbeddingResponse, LLMResponse, TokenUsage, ToolCall
from llmgate.exceptions import (
    AuthError,
    EmbeddingsNotSupported,
    LLMGateError,
    ModelNotFoundError,
    ProviderAPIError,
    RateLimitError,
)

__all__ = [
    "LLMGate",
    "LLMResponse",
    "EmbeddingResponse",
    "TokenUsage",
    "ToolCall",
    "LLMGateError",
    "AuthError",
    "RateLimitError",
    "ProviderAPIError",
    "ModelNotFoundError",
    "EmbeddingsNotSupported",
]
__version__ = "0.2.0"
