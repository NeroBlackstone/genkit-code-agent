# genkit-code-agent (gca)

A code agent speaking the [Agent Client Protocol](https://agentclientprotocol.com/) over stdio, backed by [Genkit](https://genkit.dev/).

## Build

```sh
task build   # or: go build -o gca ./cmd
```

## Run

```sh
./gca acp
```

## Model configuration

Model config lives at `~/.gca/gca.json`:

```json
{
  "provider": [
    {
      "name": "myprovider",
      "baseUrl": "https://api.openai.com/v1",
      "apiKey": "",
      "model": [
        {
          "id": "gpt-4o-mini",
          "temperature": 0.7
        },
        { "id": "gpt-4o" }
      ]
    },
    {
      "name": "deepseek",
      "baseUrl": "https://api.deepseek.com",
      "apiKey": "",
      "model": [
        { "id": "deepseek-chat" }
      ]
    }
  ]
}
```

- `provider` is an array of providers, each mapped to its own `compat_oai.OpenAICompatible` plugin: `name` becomes the model ID prefix (the full model reference is `name/id`), `baseUrl` is the OpenAI-compatible endpoint, `apiKey` authenticates the requests (empty means no auth). Names must be unique; every provider needs at least one model.
- `model` is the array of models served by that provider: `id` is the model name, `temperature` is an optional sampling setting. `max_tokens` is never sent; the endpoint's default output cap applies.
- The agent runs the first model of the first provider; the remaining entries are configured and resolvable under their `name/id` references.
- If `~/.gca/gca.json` is absent the agent runs without a model and echoes prompts; a present-but-invalid file is a startup error.

## Supported providers

Dedicated plugins under `github.com/firebase/genkit/go/plugins/compat_oai/`:

| Plugin | Models | API key | Base URL |
|---|---|---|---|
| [`openai`](https://genkit.dev/docs/go/integrations/openai/) | GPT and o-series chat models, plus text embedders | `OPENAI_API_KEY` | use `Opts` |
| [`anthropic`](https://genkit.dev/docs/go/integrations/anthropic/) | Claude via Anthropic's OpenAI-compatible endpoint | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| [`dashscope`](https://genkit.dev/docs/go/integrations/dashscope/) | Alibaba Cloud Qwen models via DashScope compatible mode | `DASHSCOPE_API_KEY` | `DASHSCOPE_BASE_URL` |
| [`deepseek`](https://genkit.dev/docs/go/integrations/deepseek/) | DeepSeek chat and reasoning models | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL` |
| [`kimi`](https://genkit.dev/docs/go/integrations/kimi/) | Moonshot AI Kimi models | `KIMI_API_KEY`, then `MOONSHOT_API_KEY` | `KIMI_BASE_URL`, then `MOONSHOT_BASE_URL` |
| [`openrouter`](https://genkit.dev/docs/go/integrations/openrouter/) | OpenRouter gateway (multi-vendor routing, fallback, reasoning controls) | `OPENROUTER_API_KEY` | `OPENROUTER_BASE_URL` |
| [`xai`](https://genkit.dev/docs/go/integrations/xai/) | xAI Grok models | `XAI_API_KEY` | `XAI_BASE_URL` |
| [`zai`](https://genkit.dev/docs/go/integrations/zai/) | Z.ai GLM models | `ZAI_API_KEY` | `ZAI_BASE_URL` |

Any other OpenAI-compatible service is reachable through the base `compat_oai.OpenAICompatible` plugin, which is what `gca.json` configures. See the [OpenAI-compatible plugin docs](https://genkit.dev/docs/go/integrations/openai-compatible/).
