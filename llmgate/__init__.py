"""llmgate — plug-and-play LLM connector via YAML config."""

from llmgate.gate import LLMGate
from llmgate.response import LLMResponse

__all__ = ["LLMGate", "LLMResponse"]
__version__ = "0.1.0"
