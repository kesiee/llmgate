# llmgate Documentation

Complete guide to using llmgate — the lightweight, YAML-configured LLM gateway for Python.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
  - [Flat Config](#flat-config)
  - [Multi-Profile Config](#multi-profile-config)
  - [Environment Variable Interpolation](#environment-variable-interpolation)
  - [Loading from .env Files](#loading-from-env-files)
- [Basic Usage](#basic-usage)
  - [Simple Chat](#simple-chat)
  - [System Prompts](#system-prompts)
  - [Multi-Turn Conversations](#multi-turn-conversations)
  - [Streaming](#streaming)
  - [Profile Switching](#profile-switching)
- [Async Support](#async-support)
  - [Async Chat](#async-chat)
  - [Async Streaming](#async-streaming)
- [Tool / Function Calling](#tool--function-calling)
  - [Defining Tools](#defining-tools)
  - [Handling Tool Calls](#handling-tool-calls)
  - [Provider-Specific Behavior](#tool-calling-provider-behavior)
- [Structured Outputs](#structured-outputs)
  - [Pydantic Models](#pydantic-models)
  - [JSON Mode](#json-mode)
  - [Handling Parse Failures](#handling-parse-failures)
- [Embeddings](#embeddings)
  - [Single and Batch](#single-and-batch-embeddings)
  - [Dimensions](#controlling-dimensions)
  - [Provider Support](#embedding-provider-support)
- [Middleware](#middleware)
  - [RetryMiddleware](#retrymiddleware)
  - [LoggingMiddleware](#loggingmiddleware)
  - [CacheMiddleware](#cachemiddleware)
  - [RateLimitMiddleware](#ratelimitmiddleware)
  - [Composing Middleware](#composing-middleware)
  - [Custom Middleware](#custom-middleware)
- [Error Handling](#error-handling)
  - [Exception Hierarchy](#exception-hierarchy)
  - [Common Patterns](#common-error-patterns)
- [Provider Setup Guides](#provider-setup-guides)
  - [OpenAI](#openai)
  - [Anthropic](#anthropic)
  - [Google Gemini](#google-gemini)
  - [Groq](#groq)
  - [Mistral](#mistral)
  - [Cohere](#cohere)
  - [DeepSeek](#deepseek)
  - [xAI (Grok)](#xai-grok)
  - [OpenRouter](#openrouter)
  - [Together AI](#together-ai)
  - [Fireworks AI](#fireworks-ai)
  - [Perplexity](#perplexity)
  - [AI21 Labs](#ai21-labs)
  - [Azure OpenAI](#azure-openai)
  - [AWS Bedrock](#aws-bedrock)
  - [Google Vertex AI](#google-vertex-ai)
  - [HuggingFace Inference](#huggingface-inference)
  - [Replicate](#replicate)
  - [NLP Cloud](#nlp-cloud)
  - [Ollama (Local)](#ollama-local)
  - [LM Studio (Local)](#lm-studio-local)
- [API Reference](#api-reference)
  - [LLMGate](#llmgate-class)
  - [LLMResponse](#llmresponse)
  - [EmbeddingResponse](#embeddingresponse)
  - [ToolCall](#toolcall)
  - [TokenUsage](#tokenusage)
- [Architecture](#architecture)
- [Adding a New Provider](#adding-a-new-provider)
- [FAQ](#faq)

---

## Overview

llmgate is a Python library that provides a single interface to 21 LLM providers. You configure your provider, model, and credentials in a YAML file, and llmgate handles the HTTP calls, response parsing, and format conversion.

**Key design principles:**

- **Minimal dependencies** — only `httpx` and `pyyaml`. No SDK lock-in.
- **YAML-driven** — swap providers by editing a config file, not code.
- **Direct HTTP** — no proxy server, no middleware by default, no overhead.
- **Provider-agnostic** — same `LLMResponse` shape regardless of provider.

**What's included in v0.2.0:**

| Feature | Description |
|---|---|
| 21 providers | OpenAI, Anthropic, Gemini, Groq, Cohere, and 16 more |
| Sync + Async | Every method has an async counterpart |
| Streaming | Real-time token streaming (sync and async) |
| Tool calling | OpenAI-style tools, auto-converted per provider |
| Structured output | Pydantic model parsing from LLM responses |
| Embeddings | Text embeddings across supported providers |
| Middleware | Retry, cache, logging, rate limiting |
| Typed exceptions | AuthError, RateLimitError, ProviderAPIError, etc. |

---

## Installation

```bash
pip install llmgt
```

> **Note:** The PyPI package name is `llmgt`, but the Python import is `llmgate`.

**Optional extras:**

```bash
pip install llmgt[aws]    # AWS Bedrock support (installs boto3)
pip install llmgt[gcp]    # Google Vertex AI support (installs google-auth)
pip install llmgt[dev]    # Development tools (pytest, pytest-mock, python-dotenv)
```

**Requirements:** Python 3.10+

---

## Configuration

llmgate reads its configuration from a YAML file. By default, it looks for `llmgate.yaml` in the current working directory.

### Flat Config

The simplest configuration — a single provider:

```yaml
# llmgate.yaml
provider: openai
model: gpt-4o
api_key: ${OPENAI_API_KEY}
temperature: 0.7
max_tokens: 1024
```

```python
from llmgate import LLMGate

gate = LLMGate()  # auto-finds llmgate.yaml in cwd
```

You can also specify a custom path:

```python
gate = LLMGate(config_path="/path/to/my-config.yaml")
```

### Multi-Profile Config

Define multiple provider configurations and switch between them:

```yaml
# llmgate.yaml
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

**How profiles work:**

1. `active_profile` sets the default profile to use.
2. `defaults` are merged into every profile (profile values take precedence).
3. You can override the active profile at init time or switch at runtime.

```python
gate = LLMGate()                        # uses "smart" (active_profile)
gate = LLMGate(profile="fast")          # override at init
gate.switch("cheap")                    # switch at runtime
```

### Environment Variable Interpolation

Any string value in the YAML can use `${ENV_VAR}` syntax:

```yaml
api_key: ${MY_API_KEY}
base_url: ${CUSTOM_ENDPOINT}
nested:
  deep:
    value: ${SOME_SECRET}
```

- Variables are resolved from `os.environ` at load time.
- Missing variables resolve to an empty string (no error).
- Works at any nesting level — strings, within dicts, within lists.

### Loading from .env Files

llmgate doesn't load `.env` files itself — use [python-dotenv](https://pypi.org/project/python-dotenv/):

```bash
pip install python-dotenv
```

```python
from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ BEFORE creating LLMGate

from llmgate import LLMGate
gate = LLMGate()
```

Example `.env` file:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...
DEEPSEEK_API_KEY=sk-...
```

---

## Basic Usage

### Simple Chat

```python
from llmgate import LLMGate

gate = LLMGate()
response = gate.chat("Explain quantum computing in one sentence")
print(response.text)
print(f"Model: {response.model}")
print(f"Provider: {response.provider}")
print(f"Tokens: {response.tokens_used}")
```

### System Prompts

Use `chat_messages()` to include a system prompt:

```python
response = gate.chat_messages([
    {"role": "system", "content": "You are a helpful coding assistant. Be concise."},
    {"role": "user", "content": "What's a closure in JavaScript?"},
])
print(response.text)
```

### Multi-Turn Conversations

Build up a conversation by appending messages:

```python
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What's a derivative?"},
]

# First turn
response = gate.chat_messages(messages)
print(response.text)

# Second turn — include the assistant's reply and a follow-up
messages.append({"role": "assistant", "content": response.text})
messages.append({"role": "user", "content": "Can you give me an example?"})

response = gate.chat_messages(messages)
print(response.text)
```

### Streaming

Get tokens as they're generated instead of waiting for the full response:

```python
# Simple prompt
for chunk in gate.stream("Write a haiku about Python"):
    print(chunk, end="", flush=True)
print()

# With full message list
messages = [
    {"role": "system", "content": "Be creative."},
    {"role": "user", "content": "Tell me a story in 3 sentences."},
]
for chunk in gate.stream_messages(messages):
    print(chunk, end="", flush=True)
print()
```

### Profile Switching

Switch providers at runtime without creating a new `LLMGate` instance:

```python
gate = LLMGate()              # uses active_profile from YAML
print(gate.provider_name)     # "anthropic"
print(gate.model)             # "claude-sonnet-4-20250514"

gate.switch("fast")
print(gate.provider_name)     # "groq"
print(gate.model)             # "llama-3.1-8b-instant"

# Call-time parameter overrides
response = gate.chat("Hello", temperature=0.2, max_tokens=50)
```

---

## Async Support

Every sync method has an async counterpart. All async methods use `httpx.AsyncClient` under the hood.

### Async Chat

```python
import asyncio
from llmgate import LLMGate

async def main():
    gate = LLMGate()

    # Simple async chat
    response = await gate.achat("Hello!")
    print(response.text)

    # With full message list
    response = await gate.achat_messages([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What's recursion?"},
    ])
    print(response.text)

asyncio.run(main())
```

### Async Streaming

```python
async def main():
    gate = LLMGate()

    async for chunk in gate.astream("Write a poem"):
        print(chunk, end="", flush=True)
    print()

    # With message list
    messages = [{"role": "user", "content": "Tell a joke"}]
    async for chunk in gate.astream_messages(messages):
        print(chunk, end="", flush=True)
    print()

asyncio.run(main())
```

**Async method mapping:**

| Sync | Async |
|---|---|
| `gate.chat()` | `await gate.achat()` |
| `gate.chat_messages()` | `await gate.achat_messages()` |
| `gate.stream()` | `async for chunk in gate.astream()` |
| `gate.stream_messages()` | `async for chunk in gate.astream_messages()` |
| `gate.embed()` | `await gate.aembed()` |

---

## Tool / Function Calling

llmgate supports tool/function calling using the OpenAI tool format. It automatically converts tools to each provider's native format.

### Defining Tools

Tools use the standard OpenAI format:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'San Francisco'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city"],
            },
        },
    }
]
```

### Handling Tool Calls

```python
response = gate.chat(
    "What's the weather in Tokyo?",
    tools=tools,
    tool_choice="auto",  # "auto", "none", or a specific tool name
)

if response.tool_calls:
    for tc in response.tool_calls:
        print(f"Function: {tc.function}")      # "get_weather"
        print(f"Arguments: {tc.arguments}")    # {"city": "Tokyo"}
        print(f"Call ID: {tc.id}")             # "call_abc123"

        # Execute the function and send the result back
        result = get_weather(**tc.arguments)  # your implementation

        # Continue the conversation with the tool result
        messages = [
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {"role": "assistant", "content": "", "tool_calls": [...]},
            {"role": "tool", "content": result, "tool_call_id": tc.id},
        ]
        final = gate.chat_messages(messages)
        print(final.text)
else:
    print(response.text)
```

### Tool Calling Provider Behavior

| Provider | Format | Notes |
|---|---|---|
| OpenAI (+ compatible) | Native OpenAI tools | Passed through directly |
| Anthropic | Converted to Anthropic tool_use | `input_schema` format |
| Gemini | Converted to functionDeclarations | Google's function calling format |

OpenAI-compatible providers (Groq, Mistral, Together, etc.) support tools natively via the inherited OpenAI format.

---

## Structured Outputs

Get validated, typed objects from LLM responses by passing a Pydantic model.

### Pydantic Models

```python
from pydantic import BaseModel
from llmgate import LLMGate

class Movie(BaseModel):
    title: str
    year: int
    genre: str
    rating: float

gate = LLMGate()
response = gate.chat(
    "Name a great sci-fi film. Respond in JSON with title, year, genre, rating.",
    response_format=Movie,
)

if response.parsed:
    movie = response.parsed
    print(f"{movie.title} ({movie.year}) - {movie.genre} - {movie.rating}/10")
else:
    # Parsing failed — raw text is still available
    print(response.text)
```

**How it works:**

1. llmgate adds `{"type": "json_object"}` to the provider request (for providers that support JSON mode).
2. The LLM responds with JSON.
3. llmgate parses the JSON and validates it against your Pydantic model.
4. The validated object is available at `response.parsed`.
5. If parsing fails, `response.parsed` is `None` and `response.text` contains the raw output.

### JSON Mode

You can also request raw JSON without Pydantic parsing:

```python
response = gate.chat(
    "List 3 colors as JSON",
    response_format={"type": "json_object"},
)
import json
data = json.loads(response.text)
```

### Handling Parse Failures

llmgate handles markdown code blocks automatically:

```python
# If the LLM wraps JSON in ```json ... ```, llmgate extracts it
response = gate.chat("Return a JSON object", response_format=MyModel)
# response.parsed works even if the LLM output was wrapped in code fences
```

If the response isn't valid JSON or doesn't match the model, `response.parsed` is `None` — no exception is raised.

---

## Embeddings

Generate text embeddings for similarity search, clustering, or RAG applications.

### Single and Batch Embeddings

```python
# Single text
result = gate.embed("Hello world")
vector = result.embeddings[0]        # list[float]
print(f"Dimensions: {len(vector)}")

# Batch
result = gate.embed(["Hello", "World", "Goodbye"])
for i, vec in enumerate(result.embeddings):
    print(f"Text {i}: {len(vec)} dimensions")

# Async
result = await gate.aembed("Hello world")
```

### Controlling Dimensions

Some providers (OpenAI, Gemini) support custom embedding dimensions:

```python
result = gate.embed("Hello", dimensions=256)
print(len(result.embeddings[0]))  # 256
```

### Embedding Provider Support

| Provider | Embedding Models | Dimensions Control |
|---|---|---|
| OpenAI (+compatible) | text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002 | ✅ |
| Gemini | text-embedding-004 | ❌ |
| Cohere | embed-english-v3.0, embed-multilingual-v3.0 | ❌ |

Providers without embedding support raise `EmbeddingsNotSupported`:

```python
from llmgate import EmbeddingsNotSupported

try:
    result = gate.embed("Hello")
except EmbeddingsNotSupported as e:
    print(f"{e.provider} doesn't support embeddings")
```

**EmbeddingResponse fields:**

```python
result.embeddings     # list[list[float]] — one vector per input
result.model          # str
result.provider       # str
result.usage          # TokenUsage(prompt_tokens, completion_tokens, total_tokens)
result.raw            # dict — full API response
```

---

## Middleware

Middleware wraps every `chat()` / `achat()` call, letting you add cross-cutting behavior without changing your application code.

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
```

Middleware is applied left-to-right: the first middleware in the list is the outermost wrapper.

### RetryMiddleware

Retries on transient errors with exponential backoff.

```python
RetryMiddleware(
    max_retries=3,          # max retry attempts (default: 3)
    backoff_factor=0.5,     # delay = factor * 2^attempt (default: 0.5)
    retryable=(             # exception types to retry on
        RateLimitError,
        ConnectionError,
        TimeoutError,
    ),
)
```

**Backoff schedule** (with default `backoff_factor=0.5`):
- Attempt 1: immediate
- Attempt 2: 0.5s delay
- Attempt 3: 1.0s delay
- Attempt 4: 2.0s delay

### LoggingMiddleware

Logs request and response details using Python's `logging` module.

```python
LoggingMiddleware(level="INFO")  # DEBUG, INFO, WARNING, ERROR
```

Log output:

```
INFO:llmgate:llmgate request: 2 message(s)
INFO:llmgate:llmgate response: provider=openai model=gpt-4o tokens=42 time=0.85s
```

### CacheMiddleware

In-memory TTL cache keyed by message content and kwargs.

```python
CacheMiddleware(ttl=300)  # cache for 5 minutes (default: 300s)
```

- Identical requests return cached responses without hitting the API.
- Different messages or kwargs are cached separately.
- Cache is in-memory and does not persist across process restarts.

### RateLimitMiddleware

Sliding-window rate limiter.

```python
RateLimitMiddleware(rpm=60)  # max 60 requests per minute
```

If the limit is reached, the middleware sleeps until a slot opens. In async mode, it uses `asyncio.sleep` so it doesn't block the event loop.

### Composing Middleware

Middleware executes left-to-right on the way in (the first middleware sees the request first), and right-to-left on the way out (the last middleware sees the response first).

Recommended order:

```python
middleware=[
    RetryMiddleware(...),       # outermost: retries everything below
    RateLimitMiddleware(...),   # rate limit before making requests
    LoggingMiddleware(...),     # log what actually goes to the API
    CacheMiddleware(...),       # innermost: cache the raw response
]
```

### Custom Middleware

Create your own middleware by subclassing `BaseMiddleware`:

```python
from llmgate.middleware import BaseMiddleware

class TimingMiddleware(BaseMiddleware):
    def handle(self, messages, call_next, **kwargs):
        import time
        start = time.monotonic()
        result = call_next(messages, **kwargs)
        elapsed = time.monotonic() - start
        print(f"Request took {elapsed:.2f}s")
        return result

    async def ahandle(self, messages, call_next, **kwargs):
        import time
        start = time.monotonic()
        result = await call_next(messages, **kwargs)
        elapsed = time.monotonic() - start
        print(f"Async request took {elapsed:.2f}s")
        return result

gate = LLMGate(middleware=[TimingMiddleware()])
```

---

## Error Handling

### Exception Hierarchy

All llmgate exceptions inherit from `LLMGateError`:

```
LLMGateError
├── AuthError              # 401/403 — bad or missing API key
├── RateLimitError         # 429 — rate or quota exceeded
├── ProviderAPIError       # other HTTP errors from the provider
├── ModelNotFoundError     # unknown model or provider
└── EmbeddingsNotSupported # provider doesn't have an embeddings API
```

Standard Python exceptions are also used:

- `FileNotFoundError` — YAML config file not found
- `ValueError` — invalid config (missing provider, unknown profile)
- `ImportError` — missing optional dependency (boto3, google-auth)

### Common Error Patterns

```python
from llmgate import (
    LLMGate,
    LLMGateError,
    AuthError,
    RateLimitError,
    ProviderAPIError,
    ModelNotFoundError,
    EmbeddingsNotSupported,
)

gate = LLMGate()

# Catch specific errors
try:
    response = gate.chat("Hello")
except AuthError as e:
    print(f"Check your API key for {e.provider}")
except RateLimitError as e:
    print(f"Slow down — rate limited by {e.provider}")
except ProviderAPIError as e:
    print(f"{e.provider} error (HTTP {e.status_code})")

# Catch-all for llmgate errors
try:
    response = gate.chat("Hello")
except LLMGateError as e:
    print(f"LLM error: {e}")

# Config errors
try:
    gate = LLMGate(config_path="nonexistent.yaml")
except FileNotFoundError:
    print("Config file not found")

try:
    gate = LLMGate(profile="nonexistent")
except ValueError as e:
    print(f"Bad config: {e}")
```

**Exception attributes:**

| Exception | Attributes |
|---|---|
| `AuthError` | `provider` |
| `RateLimitError` | `provider` |
| `ProviderAPIError` | `provider`, `status_code` |
| `ModelNotFoundError` | `model` |
| `EmbeddingsNotSupported` | `provider` |

---

## Provider Setup Guides

### OpenAI

```yaml
provider: openai
model: gpt-4o          # or gpt-4-turbo, gpt-3.5-turbo, o1-preview, etc.
api_key: ${OPENAI_API_KEY}
temperature: 0.7
max_tokens: 1024
```

Supports: chat, streaming, async, tools, embeddings, structured output.

You can also point to any OpenAI-compatible endpoint:

```yaml
provider: openai
model: my-model
api_key: ${API_KEY}
base_url: https://my-proxy.example.com/v1
```

### Anthropic

```yaml
provider: anthropic
model: claude-sonnet-4-20250514   # or claude-opus-4-20250514, claude-3-haiku, etc.
api_key: ${ANTHROPIC_API_KEY}
max_tokens: 1024
```

Supports: chat, streaming, async, tools. Does **not** support embeddings.

System messages are automatically extracted and sent via Anthropic's `system` parameter.

### Google Gemini

```yaml
provider: gemini
model: gemini-1.5-pro    # or gemini-1.5-flash, gemini-2.0-flash, etc.
api_key: ${GEMINI_API_KEY}
```

Supports: chat, streaming, async, tools, embeddings.

Messages are automatically converted to Gemini's `contents` format. System messages use `systemInstruction`.

### Groq

```yaml
provider: groq
model: llama-3.1-8b-instant   # or mixtral-8x7b-32768, etc.
api_key: ${GROQ_API_KEY}
```

OpenAI-compatible. Supports: chat, streaming, async, tools, embeddings.

### Mistral

```yaml
provider: mistral
model: mistral-large-latest   # or mistral-small-latest, open-mistral-7b, etc.
api_key: ${MISTRAL_API_KEY}
```

OpenAI-compatible. Supports: chat, streaming, async, tools, embeddings.

### Cohere

```yaml
provider: cohere
model: command-r-plus   # or command-r, command-light, etc.
api_key: ${COHERE_API_KEY}
```

Supports: chat, streaming, async, embeddings. Uses Cohere's V2 chat API.

Embedding models: `embed-english-v3.0`, `embed-multilingual-v3.0`, `embed-english-light-v3.0`.

### DeepSeek

```yaml
provider: deepseek
model: deepseek-chat   # or deepseek-coder
api_key: ${DEEPSEEK_API_KEY}
```

OpenAI-compatible. Supports: chat, streaming, async, tools, embeddings.

### xAI (Grok)

```yaml
provider: xai
model: grok-2   # or grok-beta
api_key: ${XAI_API_KEY}
```

OpenAI-compatible. Supports: chat, streaming, async, tools, embeddings.

### OpenRouter

```yaml
provider: openrouter
model: meta-llama/llama-3.1-70b-instruct
api_key: ${OPENROUTER_API_KEY}
site_url: https://myapp.com        # optional, sent as HTTP-Referer
app_name: MyApp                     # optional, sent as X-Title
```

OpenAI-compatible. Supports: chat, streaming, async, tools, embeddings.

OpenRouter gives access to hundreds of models from different providers through a single API.

### Together AI

```yaml
provider: together
model: meta-llama/Llama-3-70b-chat-hf
api_key: ${TOGETHER_API_KEY}
```

OpenAI-compatible. Supports: chat, streaming, async, tools, embeddings.

### Fireworks AI

```yaml
provider: fireworks
model: accounts/fireworks/models/llama-v3-70b-instruct
api_key: ${FIREWORKS_API_KEY}
```

OpenAI-compatible. Supports: chat, streaming, async, tools, embeddings.

### Perplexity

```yaml
provider: perplexity
model: llama-3.1-sonar-large-128k-online
api_key: ${PERPLEXITY_API_KEY}
```

OpenAI-compatible. Supports: chat, streaming, async, tools, embeddings.

### AI21 Labs

```yaml
provider: ai21
model: jamba-1.5-large   # or jamba-1.5-mini
api_key: ${AI21_API_KEY}
```

OpenAI-compatible. Supports: chat, streaming, async, tools, embeddings.

### Azure OpenAI

```yaml
provider: azure_openai
model: gpt-4o
resource_name: my-azure-resource       # your Azure resource name
deployment_name: my-gpt4o-deployment   # your deployment name
api_version: "2024-02-01"              # optional, defaults to 2024-02-01
api_key: ${AZURE_OPENAI_API_KEY}
```

Supports: chat, streaming, async, tools. Uses `api-key` header instead of Bearer token.

URL format: `https://{resource_name}.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}`

### AWS Bedrock

```bash
pip install llmgt[aws]
```

```yaml
provider: bedrock
model: anthropic.claude-3-sonnet-20240229-v1:0
region: us-east-1
```

Supports: chat, async (delegates to sync). Streaming returns full response as single chunk.

**Supported model families** (auto-detected by model ID prefix):

| Prefix | Family | Request Format |
|---|---|---|
| `anthropic.` | Claude on Bedrock | Anthropic Messages API |
| `amazon.` | Amazon Titan | Titan text generation |
| `meta.` | Meta Llama | Llama prompt format |

Requires AWS credentials via `~/.aws/credentials`, environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`), or IAM role.

### Google Vertex AI

```bash
pip install llmgt[gcp]
```

```yaml
provider: vertexai
model: gemini-1.5-pro
project_id: my-gcp-project
region: us-central1
```

Supports: chat, streaming, async. Uses Google Application Default Credentials.

Setup: `gcloud auth application-default login` or set `GOOGLE_APPLICATION_CREDENTIALS`.

### HuggingFace Inference

```yaml
provider: huggingface
model: mistralai/Mixtral-8x7B-Instruct-v0.1   # or any HF model
api_key: ${HUGGINGFACE_API_KEY}
```

Supports: chat, async. Streaming returns full response as single chunk.

**Auto-detection:** Models with "instruct", "chat", or "it" in the name use the chat completions endpoint. Others use the text generation endpoint.

### Replicate

```yaml
provider: replicate
model: meta/llama-2-70b-chat
api_key: ${REPLICATE_API_KEY}
version: "02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
```

Supports: chat, async. Uses polling (creates a prediction, then polls until complete). Streaming returns full response as single chunk.

### NLP Cloud

```yaml
provider: nlpcloud
model: chatdolphin
api_key: ${NLPCLOUD_API_KEY}
```

Supports: chat, async. Streaming returns full response as single chunk.

Multi-turn conversations are automatically converted to NLP Cloud's `input` + `history` format.

### Ollama (Local)

```yaml
provider: ollama
model: llama3.2   # or mistral, codellama, phi, etc.
```

No API key required. Runs against a local Ollama server.

```yaml
# Custom Ollama URL (default: http://localhost:11434)
provider: ollama
model: llama3.2
base_url: http://192.168.1.100:11434
```

Supports: chat, streaming, async.

### LM Studio (Local)

```yaml
provider: lmstudio
model: mistral-7b-instruct   # whatever model you have loaded
```

No API key required. Runs against LM Studio's local server (default: `http://localhost:1234`).

OpenAI-compatible. Supports: chat, streaming, async, tools, embeddings.

---

## API Reference

### LLMGate Class

```python
class LLMGate:
    def __init__(
        self,
        config_path: str | None = None,       # path to YAML config (default: ./llmgate.yaml)
        profile: str | None = None,            # override active_profile
        middleware: list[BaseMiddleware] = [],  # middleware stack
    ) -> None: ...

    # Properties
    provider_name: str       # current provider name
    model: str               # current model name
    config: dict[str, Any]   # copy of resolved config

    # Profile switching
    def switch(self, profile: str) -> None: ...

    # Sync chat
    def chat(self, prompt: str, **kwargs) -> LLMResponse: ...
    def chat_messages(self, messages: list[dict], **kwargs) -> LLMResponse: ...

    # Async chat
    async def achat(self, prompt: str, **kwargs) -> LLMResponse: ...
    async def achat_messages(self, messages: list[dict], **kwargs) -> LLMResponse: ...

    # Sync streaming
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]: ...
    def stream_messages(self, messages: list[dict], **kwargs) -> Generator[str, None, None]: ...

    # Async streaming
    async def astream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]: ...
    async def astream_messages(self, messages: list[dict], **kwargs) -> AsyncGenerator[str, None]: ...

    # Embeddings
    def embed(self, input: str | list[str], **kwargs) -> EmbeddingResponse: ...
    async def aembed(self, input: str | list[str], **kwargs) -> EmbeddingResponse: ...
```

**Common kwargs for chat/stream methods:**

| Kwarg | Type | Description |
|---|---|---|
| `temperature` | `float` | Sampling temperature (0.0-2.0) |
| `max_tokens` | `int` | Maximum tokens to generate |
| `top_p` | `float` | Nucleus sampling threshold |
| `stop` | `str \| list[str]` | Stop sequences |
| `tools` | `list[dict]` | Tool/function definitions |
| `tool_choice` | `str` | "auto", "none", or specific tool name |
| `response_format` | `type \| dict` | Pydantic model or `{"type": "json_object"}` |

### LLMResponse

```python
@dataclass
class LLMResponse:
    text: str                        # generated text
    model: str                       # model name
    provider: str                    # provider name
    tokens_used: int | None          # total tokens (if available)
    finish_reason: str | None        # "stop", "tool_calls", etc.
    raw: dict[str, Any]              # full API response
    tool_calls: list[ToolCall] = []  # tool calls (if any)
    parsed: Any = None               # Pydantic model (if response_format used)
```

`str(response)` returns `response.text`.

### EmbeddingResponse

```python
@dataclass
class EmbeddingResponse:
    embeddings: list[list[float]]   # one vector per input text
    model: str
    provider: str
    usage: TokenUsage
    raw: dict[str, Any] = {}
```

### ToolCall

```python
@dataclass
class ToolCall:
    id: str                      # unique call ID
    function: str                # function name
    arguments: dict[str, Any]    # parsed arguments
```

### TokenUsage

```python
@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
```

---

## Architecture

```
llmgate/
├── __init__.py          # public API exports
├── gate.py              # LLMGate class, provider registry
├── config.py            # YAML loading + env var interpolation
├── response.py          # LLMResponse, EmbeddingResponse, ToolCall, TokenUsage
├── exceptions.py        # typed exception hierarchy
├── middleware/
│   ├── __init__.py      # middleware exports
│   ├── base.py          # BaseMiddleware class
│   ├── retry.py         # RetryMiddleware
│   ├── logging.py       # LoggingMiddleware
│   ├── cache.py         # CacheMiddleware
│   └── ratelimit.py     # RateLimitMiddleware
└── providers/
    ├── base.py          # BaseProvider ABC
    ├── openai.py        # OpenAI (base for compatible providers)
    ├── anthropic.py     # Anthropic Claude
    ├── gemini.py        # Google Gemini
    ├── cohere.py        # Cohere
    ├── ollama.py        # Ollama (local)
    ├── azure_openai.py  # Azure OpenAI
    ├── bedrock.py       # AWS Bedrock
    ├── vertexai.py      # Google Vertex AI
    ├── huggingface.py   # HuggingFace Inference
    ├── replicate.py     # Replicate
    ├── nlpcloud.py      # NLP Cloud
    ├── groq.py          # Groq (extends OpenAI)
    ├── mistral.py       # Mistral (extends OpenAI)
    ├── deepseek.py      # DeepSeek (extends OpenAI)
    ├── together.py      # Together (extends OpenAI)
    ├── fireworks.py     # Fireworks (extends OpenAI)
    ├── perplexity.py    # Perplexity (extends OpenAI)
    ├── xai.py           # xAI/Grok (extends OpenAI)
    ├── ai21.py          # AI21 (extends OpenAI)
    ├── openrouter.py    # OpenRouter (extends OpenAI)
    └── lmstudio.py      # LM Studio (extends OpenAI)
```

**Provider inheritance:**

- `BaseProvider` — abstract class with `send()`, `stream()`, `asend()`, `astream()`, `embed()`, `aembed()`
- `OpenAIProvider` — full implementation for the OpenAI API format
- OpenAI-compatible providers (Groq, Mistral, etc.) extend `OpenAIProvider` and only override `BASE_URL` and optionally `_get_headers()` or `provider_name`

---

## Adding a New Provider

1. **Create the provider file:**

```python
# llmgate/providers/myprovider.py

from llmgate.providers.openai import OpenAIProvider
# Or: from llmgate.providers.base import BaseProvider

class MyProvider(OpenAIProvider):
    BASE_URL = "https://api.myprovider.com/v1"

    @property
    def provider_name(self) -> str:
        return "myprovider"
```

For OpenAI-compatible APIs, that's often all you need. For custom APIs, inherit from `BaseProvider` and implement `send()`, `stream()`, `asend()`, and `astream()`.

2. **Register it in `gate.py`:**

```python
PROVIDER_REGISTRY: dict[str, str] = {
    # ...existing providers...
    "myprovider": "llmgate.providers.myprovider:MyProvider",
}
```

3. **Add tests and update the README.**

---

## FAQ

**Q: Why is the PyPI package name `llmgt` instead of `llmgate`?**

The name `llmgate` was already taken on PyPI by another project. The import name remains `llmgate` — only the install command differs: `pip install llmgt`.

**Q: Can I use llmgate without a YAML file?**

Not currently — the YAML config is the primary interface. You can create a minimal YAML programmatically:

```python
import tempfile, os
from llmgate import LLMGate

with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    f.write(f"provider: openai\nmodel: gpt-4o\napi_key: {os.environ['OPENAI_API_KEY']}\n")
    gate = LLMGate(config_path=f.name)
```

**Q: Does llmgate handle retries automatically?**

Not by default. Use `RetryMiddleware` to add automatic retries:

```python
from llmgate.middleware import RetryMiddleware
gate = LLMGate(middleware=[RetryMiddleware(max_retries=3)])
```

**Q: How does llmgate compare to LiteLLM?**

llmgate is a library (you import it); LiteLLM is a proxy server. llmgate has 2 dependencies; LiteLLM has 50+. llmgate uses YAML config; LiteLLM requires code changes to switch providers. Choose llmgate if you want something lightweight and embedded in your app. Choose LiteLLM if you need a centralized proxy with a dashboard.

**Q: Can I use multiple providers in the same application?**

Yes — use multi-profile config and `gate.switch()`, or create multiple `LLMGate` instances with different configs.

**Q: Is there a timeout on requests?**

All HTTP requests have a 60-second timeout by default (set in each provider). This is not currently configurable via YAML.

**Q: What Python versions are supported?**

Python 3.10 and above.
