# llmgate

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/kesiee/llmgate.svg)](https://github.com/kesiee/llmgate/stargazers)

<!-- Uncomment after publishing to PyPI:
[![PyPI version](https://img.shields.io/pypi/v/llmgate.svg)](https://pypi.org/project/llmgate/)
[![Downloads](https://img.shields.io/pypi/dm/llmgate.svg)](https://pypi.org/project/llmgate/)
-->

Plug-and-play LLM connector via YAML config. One interface, 21 providers, zero bloat.

## Why llmgate?

You've probably seen [LiteLLM](https://github.com/BerriAI/litellm). It's great — if you want a proxy server, Redis, PostgreSQL, a dashboard, and 50+ transitive dependencies. If you just want to call an LLM from Python without installing a framework, there's nothing lightweight out there.

**llmgate** is the opposite: `pip install llmgate` pulls in exactly two dependencies (`httpx` + `pyyaml`). Drop a YAML file in your project, set your API key, and call any model. No proxy server, no database, no SDK lock-in — just a Python library that reads a config and makes HTTP calls. Swap providers by changing one line in your YAML.

| | llmgate | LiteLLM |
|---|---|---|
| Install size | ~2 MB | ~200 MB+ |
| Dependencies | 2 (`httpx`, `pyyaml`) | 50+ |
| Architecture | Library (import it) | Proxy server |
| Provider swap | Change 1 line in YAML | Change code |
| Latency overhead | ~0 (direct HTTP) | Proxy hop + DB logging |

## Install

```bash
pip install llmgate
```

Optional extras:
```bash
pip install llmgate[aws]    # AWS Bedrock (boto3)
pip install llmgate[gcp]    # Google Vertex AI (google-auth)
pip install llmgate[dev]    # pytest + dev tools
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

## System Prompts & Multi-Turn

For simple prompts use `chat()`. For system prompts or conversation history, use `chat_messages()` with the full messages list:

```python
response = gate.chat_messages([
    {"role": "system", "content": "You are a helpful coding assistant. Be concise."},
    {"role": "user", "content": "What's a closure?"},
])
print(response.text)
```

## Multi-Turn Conversations

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

```bash
pip install python-dotenv
```

```python
from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ

from llmgate import LLMGate
gate = LLMGate()  # now ${ANTHROPIC_API_KEY} etc. will resolve
```

Or use a `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...
```

See `.env.example` in the repo for all supported variables.

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

| Provider | Example Models | Env Var | Streaming | Notes |
|---|---|---|---|---|
| `openai` | gpt-4o, gpt-4-turbo | `OPENAI_API_KEY` | ✅ | |
| `anthropic` | claude-sonnet-4-20250514, claude-opus-4-20250514 | `ANTHROPIC_API_KEY` | ✅ | |
| `gemini` | gemini-1.5-pro, gemini-1.5-flash | `GEMINI_API_KEY` | ✅ | |
| `cohere` | command-r-plus, command-r | `COHERE_API_KEY` | ✅ | |
| `groq` | llama-3.1-8b-instant, mixtral-8x7b | `GROQ_API_KEY` | ✅ | OpenAI-compatible |
| `mistral` | mistral-large, mistral-small | `MISTRAL_API_KEY` | ✅ | OpenAI-compatible |
| `openrouter` | meta-llama/llama-3.1-70b-instruct | `OPENROUTER_API_KEY` | ✅ | OpenAI-compatible |
| `together` | meta-llama/Llama-3-70b-chat-hf | `TOGETHER_API_KEY` | ✅ | OpenAI-compatible |
| `fireworks` | accounts/fireworks/models/llama-v3-70b | `FIREWORKS_API_KEY` | ✅ | OpenAI-compatible |
| `perplexity` | llama-3.1-sonar-large-128k | `PERPLEXITY_API_KEY` | ✅ | OpenAI-compatible |
| `deepseek` | deepseek-chat, deepseek-coder | `DEEPSEEK_API_KEY` | ✅ | OpenAI-compatible |
| `xai` | grok-2, grok-beta | `XAI_API_KEY` | ✅ | OpenAI-compatible |
| `ai21` | jamba-1.5-large, jamba-1.5-mini | `AI21_API_KEY` | ✅ | OpenAI-compatible |
| `azure_openai` | gpt-4o (via deployment) | `AZURE_OPENAI_API_KEY` | ✅ | See [Azure setup](#azure-openai-setup) |
| `bedrock` | anthropic.claude-3, amazon.titan | AWS credentials | ❌ | See [Bedrock setup](#aws-bedrock-setup) |
| `vertexai` | gemini-1.5-pro (via Vertex) | GCP ADC | ✅ | See [Vertex setup](#google-vertex-ai-setup) |
| `huggingface` | mistralai/Mixtral-8x7B-Instruct-v0.1 | `HUGGINGFACE_API_KEY` | ❌ | Auto-detects chat models |
| `replicate` | meta/llama-2-70b-chat | `REPLICATE_API_KEY` | ❌ | Polling-based |
| `nlpcloud` | chatdolphin, finetuned-llama-3 | `NLPCLOUD_API_KEY` | ❌ | |
| `ollama` | llama3.2, mistral, codellama | none | ✅ | Local |
| `lmstudio` | any GGUF model | none | ✅ | Local, OpenAI-compatible |

Providers marked ❌ for streaming will return the full response as a single chunk when you call `stream()`.

## Error Handling

llmgate raises standard exceptions you can catch:

```python
import httpx
from llmgate import LLMGate

gate = LLMGate()

try:
    response = gate.chat("Hello")
except FileNotFoundError:
    # llmgate.yaml not found
    print("Create a llmgate.yaml config file first")
except ValueError as e:
    # Bad config: unknown provider, missing profile, missing 'provider' field
    print(f"Config error: {e}")
except httpx.HTTPStatusError as e:
    # API returned an error (401 unauthorized, 429 rate limited, 500 server error, etc.)
    print(f"API error {e.response.status_code}: {e.response.text}")
except httpx.ConnectError:
    # Can't reach the API (network issue, wrong base_url, Ollama not running)
    print("Connection failed — check your network or base_url")
except httpx.TimeoutException:
    # Request took longer than 60 seconds
    print("Request timed out")
except ImportError as e:
    # Missing optional dependency (boto3 for Bedrock, google-auth for Vertex)
    print(f"Missing dependency: {e}")
```

All API errors come through as `httpx.HTTPStatusError` with the full response body available at `e.response.text` — useful for debugging rate limits, auth issues, or quota problems.

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
pip install llmgate[aws]
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
pip install llmgate[gcp]
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

## LLMResponse

```python
response = gate.chat("Hello")
response.text           # str — the generated text
response.model          # str — model name
response.provider       # str — provider name
response.tokens_used    # int | None — total tokens
response.finish_reason  # str | None — stop reason
response.raw            # dict — full API response
```

## Async Support

Not yet — llmgate v0.1 is sync-only (`httpx` sync client). Async via `httpx.AsyncClient` is planned for v0.2. If this is blocking you, open an issue.

## Contributing

```bash
git clone https://github.com/kesiee/llmgate.git
cd llmgate
pip install -e ".[dev]"
pytest
```

The codebase is intentionally simple. Provider files live in `llmgate/providers/`. OpenAI-compatible providers inherit from `OpenAIProvider` and only override `BASE_URL` + headers. Custom providers implement `send()` and `stream()` directly.

To add a new provider:
1. Create `llmgate/providers/yourprovider.py` — inherit from `BaseProvider` (or `OpenAIProvider` if compatible)
2. Add it to `PROVIDER_REGISTRY` in `llmgate/gate.py`
3. Add a test and update this README

## License

MIT
