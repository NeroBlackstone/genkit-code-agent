package main

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"os/signal"
	"strings"

	"genkit-code-agent/agent"
	"genkit-code-agent/config"
	"genkit-code-agent/oai"

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

	var g *genkit.Genkit
	var models []agent.Model
	fullAccess := false
	if cfg, err := loadModelConfig(); err != nil {
		fmt.Fprintf(os.Stderr, "gca: %v\n", err)
		os.Exit(1)
	} else if cfg != nil {
		fullAccess = cfg.FullAccess
		wires, err := wireAll(cfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "gca: %v\n", err)
			os.Exit(1)
		}
		plugins := make([]api.Plugin, 0, len(wires))
		for _, w := range wires {
			plugins = append(plugins, w.plugin)
		}
		g = genkit.Init(ctx, genkit.WithPlugins(plugins...))

		// Every configured model becomes a value of the model config option,
		// in config order, so the first one is both a new session's default
		// and what Config.Active names. The ref is prefixed with the plugin's
		// own name, the one it registers models under: a dedicated plugin's
		// fixed name, the entry's name for the generic plugin, whose Provider
		// field is set from it.
		models = make([]agent.Model, 0, len(cfg.Provider))
		for i, p := range cfg.Provider {
			for _, m := range p.Model {
				models = append(models, agent.Model{
					Ref:    wires[i].plugin.Name() + "/" + m.ID,
					Name:   m.ID,
					Group:  p.Name,
					Config: wires[i].modelConfig(m),
				})
			}
		}
	}
	ag := agent.New(g, models)
	if fullAccess {
		ag.SetFullAccess()
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

// wireAll wires every configured provider, in order, refusing two providers
// that register the same plugin name: a dedicated plugin registers its models
// under a fixed name it does not take from the config, so a second entry for
// it would collide in genkit's registry rather than add a provider.
func wireAll(cfg *config.Config) ([]wired, error) {
	wires := make([]wired, 0, len(cfg.Provider))
	at := make(map[string]string, len(cfg.Provider))
	for _, p := range cfg.Provider {
		w, err := wire(p)
		if err != nil {
			return nil, err
		}
		if other, ok := at[w.plugin.Name()]; ok {
			return nil, fmt.Errorf("config: providers %q and %q both register the plugin %q", other, p.Name, w.plugin.Name())
		}
		at[w.plugin.Name()] = p.Name
		wires = append(wires, w)
	}
	return wires, nil
}

// wire picks the plugin the provider field names: a dedicated Genkit plugin
// for the names in pluginFor, the generic OpenAI-compatible plugin for
// "openai-compatible". There is no fallback: a name outside that set, or a
// dedicated plugin left without a usable API key, is an error rather than a
// silent detour through the generic plugin pointed at some other endpoint
// than the config asked for.
func wire(p config.Provider) (wired, error) {
	plugin, modelConfig, err := pluginFor(p)
	if err != nil {
		return wired{}, err
	}
	return wired{plugin: plugin, modelConfig: modelConfig}, nil
}

// pluginFor returns the plugin serving p and the config its models take. The
// base URL from the config is applied through Opts, which every plugin
// composes after its own default, so it wins.
func pluginFor(p config.Provider) (api.Plugin, func(config.Model) any, error) {
	opts := func() []option.RequestOption {
		if p.BaseURL == "" {
			return nil
		}
		return []option.RequestOption{option.WithBaseURL(p.BaseURL)}
	}
	// requireKey fails when p supplies no key and neither do the plugin's
	// environment variables: all of the dedicated plugins but Anthropic
	// panic at Init without one.
	requireKey := func(envs ...string) error {
		if p.APIKey != "" {
			return nil
		}
		for _, env := range envs {
			if os.Getenv(env) != "" {
				return nil
			}
		}
		return fmt.Errorf("config: provider %q: no API key: set apiKey or %s", p.Name, strings.Join(envs, ", "))
	}

	switch p.Provider {
	case "anthropic":
		// Anthropic registers its models with no key; the missing one surfaces
		// on the request rather than at Init, so no key check here.
		return &anthropic.Anthropic{APIKey: p.APIKey, Opts: opts()}, func(m config.Model) any {
			return &anthropic.ChatConfig{Temperature: m.Temperature}
		}, nil
	case "dashscope":
		if err := requireKey("DASHSCOPE_API_KEY"); err != nil {
			return nil, nil, err
		}
		return &dashscope.DashScope{APIKey: p.APIKey, Opts: opts()}, func(m config.Model) any {
			return &dashscope.ChatConfig{Temperature: m.Temperature}
		}, nil
	case "deepseek":
		if err := requireKey("DEEPSEEK_API_KEY"); err != nil {
			return nil, nil, err
		}
		// DeepSeek's thinking mode rejects any replayed assistant message
		// that lacks reasoning_content while the request carries tools,
		// which compat_oai produces whenever a turn's reasoning came back
		// empty. The middleware pads the field back in at the HTTP
		// boundary; see package oai.
		return &deepseek.DeepSeek{APIKey: p.APIKey, Opts: append(opts(),
			option.WithMiddleware(oai.PadAssistantReasoning))}, func(m config.Model) any {
			// Thinking is DeepSeek's default; sending it keeps the request
			// asking for reasoning explicitly rather than relying on that.
			return &deepseek.ChatConfig{
				Temperature: m.Temperature,
				Thinking:    &deepseek.ThinkingConfig{Type: deepseek.ThinkingTypeEnabled},
			}
		}, nil
	case "kimi":
		if err := requireKey("KIMI_API_KEY", "MOONSHOT_API_KEY"); err != nil {
			return nil, nil, err
		}
		// Kimi's config takes no temperature: the K-series drops it, so the
		// configured value has nowhere to go.
		return &kimi.Kimi{APIKey: p.APIKey, Opts: opts()}, func(config.Model) any {
			return &kimi.ChatConfig{}
		}, nil
	case "openai":
		if err := requireKey("OPENAI_API_KEY"); err != nil {
			return nil, nil, err
		}
		return &openai.OpenAI{APIKey: p.APIKey, Opts: opts()}, sdkModelConfig, nil
	case "openai-compatible":
		// A proxy behind the generic plugin needs no auth, so no key check.
		return &compat_oai.OpenAICompatible{
			Provider: p.Name,
			APIKey:   p.APIKey,
			BaseURL:  p.BaseURL,
		}, sdkModelConfig, nil
	case "openrouter":
		if err := requireKey("OPENROUTER_API_KEY"); err != nil {
			return nil, nil, err
		}
		return &openrouter.OpenRouter{APIKey: p.APIKey, Opts: opts()}, func(m config.Model) any {
			return &openrouter.ChatConfig{Temperature: m.Temperature}
		}, nil
	case "xai":
		if err := requireKey("XAI_API_KEY"); err != nil {
			return nil, nil, err
		}
		return &xai.XAI{APIKey: p.APIKey, Opts: opts()}, func(m config.Model) any {
			return &xai.ChatConfig{Temperature: m.Temperature}
		}, nil
	case "zai":
		if err := requireKey("ZAI_API_KEY"); err != nil {
			return nil, nil, err
		}
		return &zai.ZAI{APIKey: p.APIKey, Opts: opts()}, func(m config.Model) any {
			return &zai.ChatConfig{Temperature: m.Temperature}
		}, nil
	}
	return nil, nil, fmt.Errorf("config: provider %q: unknown provider %q", p.Name, p.Provider)
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
