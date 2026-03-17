# llmgate

Plug-and-play LLM connector via YAML config. One interface, 21 providers, zero bloat.

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

## Environment Variable Interpolation

Any string value in the YAML can use `${ENV_VAR}` syntax:

```yaml
api_key: ${MY_API_KEY}
base_url: ${CUSTOM_ENDPOINT}
nested:
  deep:
    value: ${SOME_SECRET}
```

Variables are resolved from `os.environ` at load time. Missing vars resolve to empty string.

## Supported Providers

| Provider | Example Models | Env Var | Notes |
|---|---|---|---|
| `openai` | gpt-4o, gpt-4-turbo | `OPENAI_API_KEY` | |
| `anthropic` | claude-sonnet-4-20250514, claude-opus-4-20250514 | `ANTHROPIC_API_KEY` | |
| `gemini` | gemini-1.5-pro, gemini-1.5-flash | `GEMINI_API_KEY` | |
| `cohere` | command-r-plus, command-r | `COHERE_API_KEY` | |
| `groq` | llama-3.1-8b-instant, mixtral-8x7b | `GROQ_API_KEY` | OpenAI-compatible |
| `mistral` | mistral-large, mistral-small | `MISTRAL_API_KEY` | OpenAI-compatible |
| `openrouter` | meta-llama/llama-3.1-70b-instruct | `OPENROUTER_API_KEY` | OpenAI-compatible |
| `together` | meta-llama/Llama-3-70b-chat-hf | `TOGETHER_API_KEY` | OpenAI-compatible |
| `fireworks` | accounts/fireworks/models/llama-v3-70b | `FIREWORKS_API_KEY` | OpenAI-compatible |
| `perplexity` | llama-3.1-sonar-large-128k | `PERPLEXITY_API_KEY` | OpenAI-compatible |
| `deepseek` | deepseek-chat, deepseek-coder | `DEEPSEEK_API_KEY` | OpenAI-compatible |
| `xai` | grok-2, grok-beta | `XAI_API_KEY` | OpenAI-compatible |
| `ai21` | jamba-1.5-large, jamba-1.5-mini | `AI21_API_KEY` | OpenAI-compatible |
| `azure_openai` | gpt-4o (via deployment) | `AZURE_OPENAI_API_KEY` | See Azure setup |
| `bedrock` | anthropic.claude-3, amazon.titan | AWS credentials | See Bedrock setup |
| `vertexai` | gemini-1.5-pro (via Vertex) | GCP ADC | See Vertex setup |
| `huggingface` | mistralai/Mixtral-8x7B-Instruct-v0.1 | `HUGGINGFACE_API_KEY` | |
| `replicate` | meta/llama-2-70b-chat | `REPLICATE_API_KEY` | Polling-based |
| `nlpcloud` | chatdolphin, finetuned-llama-3 | `NLPCLOUD_API_KEY` | |
| `ollama` | llama3.2, mistral, codellama | none | Local |
| `lmstudio` | any GGUF model | none | Local, OpenAI-compatible |

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

Requires AWS credentials configured via `~/.aws/credentials`, env vars, or IAM role. Supports Anthropic Claude, Amazon Titan, and Meta Llama model families on Bedrock.

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

## License

MIT
