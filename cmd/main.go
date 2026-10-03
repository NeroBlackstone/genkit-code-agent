package main

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"os/signal"

	"genkit-code-agent/agent"
	"genkit-code-agent/config"

	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/firebase/genkit/go/plugins/compat_oai/anthropic"
	"github.com/firebase/genkit/go/plugins/compat_oai/dashscope"
	"github.com/firebase/genkit/go/plugins/compat_oai/deepseek"
	"github.com/firebase/genkit/go/plugins/compat_oai/kimi"
	"github.com/firebase/genkit/go/plugins/compat_oai/openai"
	"github.com/firebase/genkit/go/plugins/compat_oai/openrouter"
	"github.com/firebase/genkit/go/plugins/compat_oai/xai"
	"github.com/firebase/genkit/go/plugins/compat_oai/zai"
	openaiGo "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

const usage = `usage: gca <command>

commands:
  acp    start the agent over stdio using the Agent Client Protocol
  help   show this help
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}

	switch os.Args[1] {
	case "acp":
		runACP()
	case "help", "-h":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "gca: unknown command %q\n\n%s", os.Args[1], usage)
		os.Exit(2)
	}
}

func runACP() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	ag := agent.New()
	if cfg, err := loadModelConfig(); err != nil {
		fmt.Fprintf(os.Stderr, "gca: %v\n", err)
		os.Exit(1)
	} else if cfg != nil {
		wires := make([]wired, 0, len(cfg.Provider))
		plugins := make([]api.Plugin, 0, len(cfg.Provider))
		for _, p := range cfg.Provider {
			w := wire(p)
			wires = append(wires, w)
			plugins = append(plugins, w.plugin)
		}
		g := genkit.Init(ctx, genkit.WithPlugins(plugins...))

		// Every configured model becomes a value of the model config option,
		// in config order, so the first one is both a new session's default
		// and what Config.Active names.
		models := make([]agent.Model, 0, len(cfg.Provider))
		for i, p := range cfg.Provider {
			for _, m := range p.Model {
				models = append(models, agent.Model{
					Ref:    p.Name + "/" + m.ID,
					Name:   m.ID,
					Group:  p.Name,
					Config: wires[i].modelConfig(m),
				})
			}
		}
		ag.SetModels(g, models)
	}

	conn := acp.NewAgentSideConnection(ag, os.Stdout, os.Stdin)
	conn.SetLogger(slog.Default())
	ag.SetAgentConnection(conn)

	select {
	case <-conn.Done():
		fmt.Fprintln(os.Stderr, "gca: client disconnected")
	case <-ctx.Done():
	}
}

// loadModelConfig reads ~/.gca/gca.json. A missing file is not an error: the
// agent falls back to echoing prompts. A present-but-invalid file is fatal.
func loadModelConfig() (*config.Config, error) {
	path, err := config.DefaultPath()
	if err != nil {
		return nil, err
	}
	cfg, err := config.Load(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	return cfg, err
}

// wired pairs a configured provider with the plugin that serves it and the
// per-model request config that plugin validates.
type wired struct {
	plugin      api.Plugin
	modelConfig func(config.Model) any
}

// wire picks the plugin for a provider: the dedicated Genkit plugin for the
// names in dedicated(), the generic OpenAI-compatible plugin for anything else
// and for a dedicated provider configured without a usable API key, since all
// of them but Anthropic panic at Init without one.
func wire(p config.Provider) wired {
	if plugin, modelConfig := dedicated(p); plugin != nil {
		return wired{plugin: plugin, modelConfig: modelConfig}
	}
	return wired{
		plugin: &compat_oai.OpenAICompatible{
			Provider: p.Name,
			APIKey:   p.APIKey,
			BaseURL:  p.BaseURL,
		},
		modelConfig: sdkModelConfig,
	}
}

// dedicated returns the dedicated plugin covering p and the config its models
// take, or nil when p.Name has no dedicated plugin or p supplies no key and the
// plugin's environment variables are empty too. The base URL from the config
// is applied through Opts, which every plugin composes after its own default,
// so it wins.
func dedicated(p config.Provider) (api.Plugin, func(config.Model) any) {
	opts := func() []option.RequestOption {
		if p.BaseURL == "" {
			return nil
		}
		return []option.RequestOption{option.WithBaseURL(p.BaseURL)}
	}
	hasKey := func(envs ...string) bool {
		if p.APIKey != "" {
			return true
		}
		for _, env := range envs {
			if os.Getenv(env) != "" {
				return true
			}
		}
		return false
	}

	switch p.Name {
	case "anthropic":
		// Anthropic registers its models with no key; the missing one surfaces
		// on the request rather than at Init, so no key check here.
		return &anthropic.Anthropic{APIKey: p.APIKey, Opts: opts()}, func(m config.Model) any {
			return &anthropic.ChatConfig{Temperature: m.Temperature}
		}
	case "dashscope":
		if !hasKey("DASHSCOPE_API_KEY") {
			return nil, nil
		}
		return &dashscope.DashScope{APIKey: p.APIKey, Opts: opts()}, func(m config.Model) any {
			return &dashscope.ChatConfig{Temperature: m.Temperature}
		}
	case "deepseek":
		if !hasKey("DEEPSEEK_API_KEY") {
			return nil, nil
		}
		return &deepseek.DeepSeek{APIKey: p.APIKey, Opts: opts()}, func(m config.Model) any {
			// Thinking is DeepSeek's default; sending it keeps the request
			// asking for reasoning explicitly rather than relying on that.
			return &deepseek.ChatConfig{
				Temperature: m.Temperature,
				Thinking:    &deepseek.ThinkingConfig{Type: deepseek.ThinkingTypeEnabled},
			}
		}
	case "kimi":
		if !hasKey("KIMI_API_KEY", "MOONSHOT_API_KEY") {
			return nil, nil
		}
		// Kimi's config takes no temperature: the K-series drops it, so the
		// configured value has nowhere to go.
		return &kimi.Kimi{APIKey: p.APIKey, Opts: opts()}, func(config.Model) any {
			return &kimi.ChatConfig{}
		}
	case "openai":
		if !hasKey("OPENAI_API_KEY") {
			return nil, nil
		}
		return &openai.OpenAI{APIKey: p.APIKey, Opts: opts()}, sdkModelConfig
	case "openrouter":
		if !hasKey("OPENROUTER_API_KEY") {
			return nil, nil
		}
		return &openrouter.OpenRouter{APIKey: p.APIKey, Opts: opts()}, func(m config.Model) any {
			return &openrouter.ChatConfig{Temperature: m.Temperature}
		}
	case "xai":
		if !hasKey("XAI_API_KEY") {
			return nil, nil
		}
		return &xai.XAI{APIKey: p.APIKey, Opts: opts()}, func(m config.Model) any {
			return &xai.ChatConfig{Temperature: m.Temperature}
		}
	case "zai":
		if !hasKey("ZAI_API_KEY") {
			return nil, nil
		}
		return &zai.ZAI{APIKey: p.APIKey, Opts: opts()}, func(m config.Model) any {
			return &zai.ChatConfig{Temperature: m.Temperature}
		}
	}
	return nil, nil
}

// sdkModelConfig builds the OpenAI SDK request params the generic plugin, and
// the OpenAI plugin, validate a model's config against.
func sdkModelConfig(m config.Model) any {
	params := &openaiGo.ChatCompletionNewParams{}
	if m.Temperature != nil {
		params.Temperature = openaiGo.Float(*m.Temperature)
	}
	return params
}
