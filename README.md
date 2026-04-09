# llmgate

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?v=2)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg?v=2)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/kesiee/llmgate.svg?v=2)](https://github.com/kesiee/llmgate/stargazers)
[![Tests](https://github.com/kesiee/llmgate/actions/workflows/tests.yml/badge.svg)](https://github.com/kesiee/llmgate/actions/workflows/tests.yml)

[![PyPI version](https://img.shields.io/pypi/v/llmgt.svg?v=2)](https://pypi.org/project/llmgt/)
[![Downloads](https://img.shields.io/pypi/dm/llmgt.svg?v=2)](https://pypi.org/project/llmgt/)

Plug-and-play LLM connector via YAML config. One interface, 21 providers, zero bloat.

## Why llmgate?

You've probably seen [LiteLLM](https://github.com/BerriAI/litellm). It's great — if you want a proxy server, Redis, PostgreSQL, a dashboard, and 50+ transitive dependencies. If you just want to call an LLM from Python without installing a framework, there's nothing lightweight out there.

**llmgate** is the opposite: `pip install llmgt` pulls in exactly two dependencies (`httpx` + `pyyaml`). Drop a YAML file in your project, set your API key, and call any model. No proxy server, no database, no SDK lock-in — just a Python library that reads a config and makes HTTP calls. Swap providers by changing one line in your YAML.

| | llmgate | LiteLLM |
|---|---|---|
| Install size | ~2 MB | ~200 MB+ |
| Dependencies | 2 (`httpx`, `pyyaml`) | 50+ |
| Architecture | Library (import it) | Proxy server |
| Provider swap | Change 1 line in YAML | Change code |
| Latency overhead | ~0 (direct HTTP) | Proxy hop + DB logging |

> **Note:** The PyPI package is `llmgt` (`pip install llmgt`), but the import is `llmgate`.

## Install

```bash
pip install llmgt
```

Optional extras:
```bash
pip install llmgt[aws]    # AWS Bedrock (boto3)
pip install llmgt[gcp]    # Google Vertex AI (google-auth)
pip install llmgt[dev]    # pytest + dev tools
```

## Quickstart

1. Create `llmgate.yaml` in your project:

```yaml
provider: anthropic
model: claude-sonnet-4-20250514
api_key: ${ANTHROPIC_API_KEY}
temperature: 0.7
max_tokens: 1024
```

2. Use it:

```python
from llmgate import LLMGate

gate = LLMGate()
response = gate.chat("Explain transformers in one sentence")
print(response.text)
print(response.tokens_used)

# Streaming
for chunk in gate.stream("Write a haiku"):
    print(chunk, end="", flush=True)
```

## Async Support

Every method has an async counterpart powered by `httpx.AsyncClient`:

```python
import asyncio
from llmgate import LLMGate

async def main():
    gate = LLMGate()

    # Async chat
    response = await gate.achat("Hello!")
    print(response.text)

    # Async streaming
    async for chunk in gate.astream("Write a haiku"):
        print(chunk, end="", flush=True)

    # Async with full message list
    response = await gate.achat_messages([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What's a closure?"},
    ])

asyncio.run(main())
```

## System Prompts & Multi-Turn

For simple prompts use `chat()`. For system prompts or conversation history, use `chat_messages()` with the full messages list:

```python
response = gate.chat_messages([
    {"role": "system", "content": "You are a helpful coding assistant. Be concise."},
    {"role": "user", "content": "What's a closure?"},
])
print(response.text)
```

Build up conversation history and pass it in:

```python
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What's a derivative?"},
]

response = gate.chat_messages(messages)
print(response.text)

# Continue the conversation
messages.append({"role": "assistant", "content": response.text})
messages.append({"role": "user", "content": "Can you give me an example?"})

response = gate.chat_messages(messages)
print(response.text)
```

Streaming works with full message lists too:

```python
for chunk in gate.stream_messages(messages):
    print(chunk, end="", flush=True)
```

## Tool / Function Calling

Pass OpenAI-style tool definitions — llmgate automatically converts them to each provider's native format:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

response = gate.chat("What's the weather in NYC?", tools=tools, tool_choice="auto")

if response.tool_calls:
    for tc in response.tool_calls:
        print(f"{tc.function}({tc.arguments})")
        # get_weather({'city': 'NYC'})
```

Supported providers: OpenAI, Anthropic, Gemini (and all OpenAI-compatible providers).

## Structured Outputs

Pass a Pydantic model as `response_format` to get a validated, typed object:

```python
from pydantic import BaseModel

class Movie(BaseModel):
    title: str
    year: int
    rating: float

response = gate.chat(
    "Name a great sci-fi film. Respond in JSON.",
    response_format=Movie,
)
movie = response.parsed  # Movie(title='Inception', year=2010, rating=8.8)
```

## Embeddings

Generate embeddings with providers that support them:

```python
# Single text
result = gate.embed("Hello world")
vector = result.embeddings[0]  # list[float]

# Batch
result = gate.embed(["Hello", "World"])
vectors = result.embeddings  # list[list[float]]

# With dimensions (OpenAI)
result = gate.embed("Hello", dimensions=256)

# Async
result = await gate.aembed("Hello world")
```

Supported providers: OpenAI (+ compatible), Gemini, Cohere. Others raise `EmbeddingsNotSupported`.

## Middleware

Compose middleware for retry, logging, caching, and rate limiting:

```python
from llmgate import LLMGate
from llmgate.middleware import (
    RetryMiddleware,
    LoggingMiddleware,
    CacheMiddleware,
    RateLimitMiddleware,
)

gate = LLMGate(middleware=[
    RetryMiddleware(max_retries=3, backoff_factor=0.5),
    LoggingMiddleware(level="INFO"),
    CacheMiddleware(ttl=300),
    RateLimitMiddleware(rpm=60),
])

response = gate.chat("Hello")  # retries, logs, caches, rate-limits

# Works with async too
response = await gate.achat("Hello")
```

## Multi-Profile Config

```yaml
active_profile: smart

defaults:
  temperature: 0.7
  max_tokens: 1024

profiles:
  smart:
    provider: anthropic
    model: claude-sonnet-4-20250514
    api_key: ${ANTHROPIC_API_KEY}

  fast:
    provider: groq
    model: llama-3.1-8b-instant
    api_key: ${GROQ_API_KEY}

  cheap:
    provider: deepseek
    model: deepseek-chat
    api_key: ${DEEPSEEK_API_KEY}

  local:
    provider: ollama
    model: llama3.2
```

Hot-swap profiles at runtime:

```python
gate = LLMGate()                          # uses "smart" profile
gate.switch("fast")                       # swap to Groq
response = gate.chat("Hello", temperature=0.2)  # call-time overrides
```

## Loading API Keys from .env

llmgate resolves `${ENV_VAR}` from `os.environ`. To load keys from a `.env` file, use [python-dotenv](https://pypi.org/project/python-dotenv/):

```python
from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ

from llmgate import LLMGate
gate = LLMGate()  # now ${ANTHROPIC_API_KEY} etc. will resolve
```

## Environment Variable Interpolation

Any string value in the YAML can use `${ENV_VAR}` syntax — not just `api_key`:

```yaml
api_key: ${MY_API_KEY}
base_url: ${CUSTOM_ENDPOINT}
nested:
  deep:
    value: ${SOME_SECRET}
```

Variables are resolved from `os.environ` at load time. Missing vars resolve to empty string.

## Supported Providers

| Provider | Example Models | Env Var | Streaming | Embeddings | Tool Calling |
|---|---|---|---|---|---|
| `openai` | gpt-4o, gpt-4-turbo | `OPENAI_API_KEY` | ✅ | ✅ | ✅ |
| `anthropic` | claude-sonnet-4-20250514, claude-opus-4-20250514 | `ANTHROPIC_API_KEY` | ✅ | ❌ | ✅ |
| `gemini` | gemini-1.5-pro, gemini-1.5-flash | `GEMINI_API_KEY` | ✅ | ✅ | ✅ |
| `cohere` | command-r-plus, command-r | `COHERE_API_KEY` | ✅ | ✅ | ❌ |
| `groq` | llama-3.1-8b-instant, mixtral-8x7b | `GROQ_API_KEY` | ✅ | ✅ | ✅ |
| `mistral` | mistral-large, mistral-small | `MISTRAL_API_KEY` | ✅ | ✅ | ✅ |
| `openrouter` | meta-llama/llama-3.1-70b-instruct | `OPENROUTER_API_KEY` | ✅ | ✅ | ✅ |
| `together` | meta-llama/Llama-3-70b-chat-hf | `TOGETHER_API_KEY` | ✅ | ✅ | ✅ |
| `fireworks` | accounts/fireworks/models/llama-v3-70b | `FIREWORKS_API_KEY` | ✅ | ✅ | ✅ |
| `perplexity` | llama-3.1-sonar-large-128k | `PERPLEXITY_API_KEY` | ✅ | ✅ | ✅ |
| `deepseek` | deepseek-chat, deepseek-coder | `DEEPSEEK_API_KEY` | ✅ | ✅ | ✅ |
| `xai` | grok-2, grok-beta | `XAI_API_KEY` | ✅ | ✅ | ✅ |
| `ai21` | jamba-1.5-large, jamba-1.5-mini | `AI21_API_KEY` | ✅ | ✅ | ✅ |
| `azure_openai` | gpt-4o (via deployment) | `AZURE_OPENAI_API_KEY` | ✅ | ❌ | ✅ |
| `bedrock` | anthropic.claude-3, amazon.titan | AWS credentials | ❌ | ❌ | ❌ |
| `vertexai` | gemini-1.5-pro (via Vertex) | GCP ADC | ✅ | ❌ | ❌ |
| `huggingface` | mistralai/Mixtral-8x7B-Instruct-v0.1 | `HUGGINGFACE_API_KEY` | ❌ | ❌ | ❌ |
| `replicate` | meta/llama-2-70b-chat | `REPLICATE_API_KEY` | ❌ | ❌ | ❌ |
| `nlpcloud` | chatdolphin, finetuned-llama-3 | `NLPCLOUD_API_KEY` | ❌ | ❌ | ❌ |
| `ollama` | llama3.2, mistral, codellama | none | ✅ | ❌ | ❌ |
| `lmstudio` | any GGUF model | none | ✅ | ✅ | ✅ |

Providers marked ❌ for streaming return the full response as a single chunk. OpenAI-compatible providers (Groq, Mistral, etc.) inherit embeddings and tool calling via the OpenAI API format.

## Error Handling

llmgate provides typed exceptions for common failure modes:

```python
from llmgate import LLMGate, AuthError, RateLimitError, ProviderAPIError

gate = LLMGate()

try:
    response = gate.chat("Hello")
except FileNotFoundError:
    print("Create a llmgate.yaml config file first")
except AuthError as e:
    print(f"Bad API key for {e.provider}")
except RateLimitError as e:
    print(f"Rate limited by {e.provider} — back off and retry")
except ProviderAPIError as e:
    print(f"{e.provider} returned HTTP {e.status_code}")
except ValueError as e:
    print(f"Config error: {e}")
```

Full exception hierarchy:

| Exception | When |
|---|---|
| `AuthError` | 401/403 — bad or missing API key |
| `RateLimitError` | 429 — rate or quota exceeded |
| `ProviderAPIError` | Other HTTP errors from the provider |
| `ModelNotFoundError` | Unknown model or provider |
| `EmbeddingsNotSupported` | Provider doesn't offer embeddings |

All inherit from `LLMGateError` for catch-all handling.

## LLMResponse

```python
response = gate.chat("Hello")
response.text           # str — the generated text
response.model          # str — model name
response.provider       # str — provider name
response.tokens_used    # int | None — total tokens
response.finish_reason  # str | None — stop reason
response.tool_calls     # list[ToolCall] — tool/function calls
response.parsed         # BaseModel | None — structured output
response.raw            # dict — full API response
```

## Azure OpenAI Setup

```yaml
profiles:
  azure:
    provider: azure_openai
    model: gpt-4o
    resource_name: my-azure-resource
    deployment_name: my-gpt4o-deployment
    api_version: "2024-02-01"
    api_key: ${AZURE_OPENAI_API_KEY}
```

## AWS Bedrock Setup

```bash
pip install llmgt[aws]
```

```yaml
profiles:
  aws:
    provider: bedrock
    model: anthropic.claude-3-sonnet-20240229-v1:0  # or amazon.titan-*, meta.*
    region: us-east-1
```

Requires AWS credentials configured via `~/.aws/credentials`, env vars, or IAM role. Supports Anthropic Claude, Amazon Titan, and Meta Llama model families on Bedrock — detected automatically by model ID prefix.

## Google Vertex AI Setup

```bash
pip install llmgt[gcp]
```

```yaml
profiles:
  gcp:
    provider: vertexai
    model: gemini-1.5-pro
    project_id: my-gcp-project
    region: us-central1
```

Uses Google Application Default Credentials. Run `gcloud auth application-default login` or set `GOOGLE_APPLICATION_CREDENTIALS`.

## Contributing

```bash
git clone https://github.com/kesiee/llmgate.git
cd llmgate
pip install -e ".[dev]"
pytest
```

The codebase is intentionally simple. Provider files live in `llmgate/providers/`. OpenAI-compatible providers inherit from `OpenAIProvider` and only override `BASE_URL` + headers. Custom providers implement `send()`, `stream()`, `asend()`, and `astream()` directly.

To add a new provider:
1. Create `llmgate/providers/yourprovider.py` — inherit from `BaseProvider` (or `OpenAIProvider` if compatible)
2. Add it to `PROVIDER_REGISTRY` in `llmgate/gate.py`
3. Add a test and update this README

## License

MIT
