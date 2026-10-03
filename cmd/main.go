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
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go"
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
	case "help", "-h", "--help":
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
		plugins := make([]api.Plugin, 0, len(cfg.Provider))
		for _, p := range cfg.Provider {
			plugins = append(plugins, &compat_oai.OpenAICompatible{
				Provider: p.Name,
				APIKey:   p.APIKey,
				BaseURL:  p.BaseURL,
			})
		}
		g := genkit.Init(ctx, genkit.WithPlugins(plugins...))
		p, m := cfg.Active()
		ag.SetModel(g, ai.NewModelRef(p.Name+"/"+m.ID, modelParams(m)))
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

func modelParams(m config.Model) *openai.ChatCompletionNewParams {
	params := &openai.ChatCompletionNewParams{}
	if m.Temperature != nil {
		params.Temperature = openai.Float(*m.Temperature)
	}
	return params
}
