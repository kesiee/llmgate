"""llmgate quickstart example."""

from llmgate import LLMGate

# Auto-finds llmgate.yaml in current directory
gate = LLMGate()

# Simple chat
response = gate.chat("Explain transformers in one sentence")
print(response.text)
print(f"Tokens: {response.tokens_used}")
print(f"Model: {response.model}")

# Streaming
print("\nStreaming:")
for chunk in gate.stream("Write a haiku about coding"):
    print(chunk, end="", flush=True)
print()

# Switch profile
gate.switch("fast")
response = gate.chat("Hello!", temperature=0.2)
print(f"\n{response.provider}/{response.model}: {response.text}")
