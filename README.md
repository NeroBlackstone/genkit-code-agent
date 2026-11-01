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
      "name": "myproxy",
      "provider": "openai-compatible",
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
      "name": "ds",
      "provider": "deepseek",
      "baseUrl": "https://api.deepseek.com",
      "apiKey": "",
      "model": [
        { "id": "deepseek-chat" }
      ]
    }
  ]
}
```

- `provider` is an array of providers. `name` only identifies an entry — it is the group header its models show under in the model selector and, for `openai-compatible`, the prefix on its model references (`name/id`); a dedicated plugin's references carry the plugin's own name instead. `provider` names the plugin that serves the entry: one of the names in the table below, or `openai-compatible` for the base `compat_oai.OpenAICompatible` plugin. There is no fallback — an unknown name, or a dedicated plugin configured without an API key while its key environment variables are empty, is a startup error, since the dedicated plugins panic at startup with no key and a typo would otherwise be served by a different plugin than the config asked for. `baseUrl` is the endpoint the plugin is pointed at, `apiKey` authenticates the requests (empty means no auth; `openai-compatible` needs none). Names must be unique, every provider needs at least one model, and two entries may not name the same dedicated plugin — it registers its models under a fixed name, so a second entry would collide rather than add a provider (`openai-compatible` may repeat, one plugin per entry).
- `model` is the array of models served by that provider: `id` is the model name, `temperature` is an optional sampling setting, applied through the provider's own config type (Kimi's has no temperature field, so it is dropped there). DeepSeek requests also carry `thinking: {"type": "enabled"}` so reasoning is asked for explicitly. `max_tokens` is never sent; the endpoint's default output cap applies.
- Every configured model is exposed to the client as an ACP session config option (`category: "model"`, grouped by the entry's `name`). The client switches with `session/set_config_option`, which changes the model for that session only; a new session starts on the first model of the first provider, what `Config.Active` names.
- If `~/.gca/gca.json` is absent the agent runs without a model and echoes prompts; a present-but-invalid file is a startup error.

## Sessions

A conversation lives in the store, not in the process: `~/.gca/workspaces/<cwd hash>/sessions/<id>.json` holds the whole transcript as one JSON envelope, and each turn reads it, sends it to the model, and writes it back with the new exchange appended. What `gca` keeps in memory is only what a running process needs — the session id's workspace, its model selection, and the handle to a turn in flight — so nothing has to be rebuilt after a restart: an id from before the restart resumes with its model and its conversation.

Both `session/resume` (continue without a replay) and `session/load` (stream the stored conversation back as notifications) are served from that file. A workspace that cannot be opened leaves sessions running turn by turn with no history, and a stored file that cannot be read is quarantined aside as `.bak` rather than overwritten.

## File tools

File access comes from Genkit's `Filesystem` middleware, not from the client's `fs/read_text_file` / `fs/write_text_file` methods — ACP v2 removes those, and this agent never calls them whatever capabilities the client advertises. Each turn builds the middleware once, rooted at the session's working directory: `list_files` and `read_file` are always registered, `write_file` and `edit_file` join them since write access is enabled, and `os.Root` keeps every path inside the root — `..`, absolute paths, and symlinks that climb out are refused.

`read_file` answers with a short summary and delivers the file itself as a user message on the next model turn; `edit_file`, and `write_file` over an existing file, refuse work on a file that was not read first, and refuse it again when the file changed on disk since that read. The middleware instance lives for the whole turn, which is what lets that read-before-write hold across the approval gate: an approved `edit_file` runs in a later Generate of the same turn and has to find the state the approved `read_file` recorded in an earlier one. A client that wants to expose more than one directory, unsaved editor state, or command execution should put an MCP server on the session (`mcpServers` on `session/new` / `session/resume`), where those tools sit on the same footing as the rest.

## Tool approval

Every tool call is gated: the model's request is held until the client answers `session/request_permission`, which the agent asks with the four outcomes the protocol defines — allow once, always allow, reject once, always reject. The two `once` choices resolve only the call they answer; the two `always` choices are remembered on the session, in memory only — nothing is written to the store or to `gca.json`, so a new session or a restarted agent asks again.

A held call stays `pending` under the `toolCallId` its creating update was filed with, then runs to `completed` — or closes as `failed` with the refusal — under that same id once the answer arrives. A refusal is answered to the model as a tool response rather than left hanging, so the turn still finishes on a complete conversation, and a `cancelled` outcome ends the prompt turn with `stopReason: "cancelled"`, storing nothing. A client that never answers `session/request_permission` fails the turn rather than letting a tool run unasked.

## Skills

A `skills/` directory against the agent's process working directory holds one skill per direct subdirectory containing a `SKILL.md`. Only each skill's directory name and its frontmatter `description` are listed, in a `<skills>` block appended to the system prompt; the body loads on demand through the `use_skill` tool the model calls, so the heavy instructions stay off the hot path. Discovery is one level deep — nested directories and names starting with `.` are skipped — and a directory that cannot be read is skipped with a warning rather than failing the turn, so a missing `skills/` directory means no skills and no error.

`use_skill` is pre-approved: loading a skill reads a local file, so the approval gate never asks about it. It is still reported to the client as a tool call like any other, titled after the skill it loads. The loaded body rides in the transcript, so it stays in the model's context for the rest of the session.

## Supported providers

The `provider` field takes the name of a dedicated plugin under `github.com/firebase/genkit/go/plugins/compat_oai/`:

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

Any other OpenAI-compatible service is reachable through the base `compat_oai.OpenAICompatible` plugin, selected with `"provider": "openai-compatible"`. See the [OpenAI-compatible plugin docs](https://genkit.dev/docs/go/integrations/openai-compatible/).
