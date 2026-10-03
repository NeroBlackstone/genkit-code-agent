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

- `provider` is an array of providers. A `name` from the table below is served by that provider's dedicated plugin; any other name, and a named provider configured without an API key while its key environment variables are empty, is served by the base `compat_oai.OpenAICompatible` plugin, since the dedicated plugins panic at startup with no key. `name` is the model ID prefix (the full model reference is `name/id`), `baseUrl` is the endpoint the plugin is pointed at, `apiKey` authenticates the requests (empty means no auth). Names must be unique; every provider needs at least one model.
- `model` is the array of models served by that provider: `id` is the model name, `temperature` is an optional sampling setting, applied through the provider's own config type (Kimi's has no temperature field, so it is dropped there). DeepSeek requests also carry `thinking: {"type": "enabled"}` so reasoning is asked for explicitly. `max_tokens` is never sent; the endpoint's default output cap applies.
- Every configured model is exposed to the client as an ACP session config option (`category: "model"`, grouped by provider). The client switches with `session/set_config_option`, which changes the model for that session only; a new session starts on the first model of the first provider, what `Config.Active` names.
- If `~/.gca/gca.json` is absent the agent runs without a model and echoes prompts; a present-but-invalid file is a startup error.

## Sessions

Sessions live in memory only. The agent advertises `session/resume` (not `session/load`): a client reopening a thread names the session id, and an id it does not know is registered on the spot with the default model, so prompts continue without any history replay. Nothing survives a restart of `gca`; `session/load` with a replayed conversation is left for when sessions are persisted.

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
