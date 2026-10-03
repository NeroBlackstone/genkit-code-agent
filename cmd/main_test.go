package main

import (
	"testing"

	"genkit-code-agent/config"

	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/firebase/genkit/go/plugins/compat_oai/deepseek"
	"github.com/firebase/genkit/go/plugins/compat_oai/kimi"
	openaiGo "github.com/openai/openai-go"
)

func clearKeyEnvs(t *testing.T) {
	t.Helper()
	for _, env := range []string{
		"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DASHSCOPE_API_KEY",
		"DEEPSEEK_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY",
		"OPENROUTER_API_KEY", "XAI_API_KEY", "ZAI_API_KEY",
	} {
		t.Setenv(env, "")
	}
}

func provider(name string) config.Provider {
	return config.Provider{Name: name, APIKey: "k", Model: []config.Model{{ID: "m"}}}
}

func TestWireDedicatedPlugins(t *testing.T) {
	clearKeyEnvs(t)
	for _, name := range []string{"openai", "anthropic", "dashscope", "deepseek", "kimi", "openrouter", "xai", "zai"} {
		if _, ok := wire(provider(name)).plugin.(*compat_oai.OpenAICompatible); ok {
			t.Errorf("wire(%q) served by the generic plugin, want the dedicated one", name)
		}
	}
}

func TestWireFallsBackToGeneric(t *testing.T) {
	clearKeyEnvs(t)
	cases := []struct {
		name string
		p    config.Provider
	}{
		{"unknown provider", config.Provider{Name: "myproxy", Model: []config.Model{{ID: "m"}}}},
		{"no key anywhere", config.Provider{Name: "deepseek", Model: []config.Model{{ID: "m"}}}},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			w := wire(tt.p)
			if _, ok := w.plugin.(*compat_oai.OpenAICompatible); !ok {
				t.Fatalf("plugin = %T, want *compat_oai.OpenAICompatible", w.plugin)
			}
			if _, ok := w.modelConfig(config.Model{ID: "m"}).(*openaiGo.ChatCompletionNewParams); !ok {
				t.Fatalf("model config = %T, want *openaiGo.ChatCompletionNewParams", w.modelConfig(config.Model{ID: "m"}))
			}
		})
	}
}

func TestWireKeyFromEnvironment(t *testing.T) {
	clearKeyEnvs(t)
	t.Setenv("DEEPSEEK_API_KEY", "from-env")
	if _, ok := wire(provider("deepseek")).plugin.(*deepseek.DeepSeek); !ok {
		t.Fatal("deepseek with a key only in the environment is not served by its plugin")
	}
}

func TestModelConfig(t *testing.T) {
	clearKeyEnvs(t)
	temp := 0.7

	if cfg, ok := wire(provider("deepseek")).modelConfig(config.Model{ID: "m", Temperature: &temp}).(*deepseek.ChatConfig); !ok {
		t.Fatalf("deepseek config = %T, want *deepseek.ChatConfig", cfg)
	} else {
		if cfg.Temperature == nil || *cfg.Temperature != temp {
			t.Errorf("temperature = %v, want %v", cfg.Temperature, temp)
		}
		if cfg.Thinking == nil || cfg.Thinking.Type != deepseek.ThinkingTypeEnabled {
			t.Errorf("thinking = %+v, want enabled", cfg.Thinking)
		}
	}

	// Kimi's config has no temperature field to carry it.
	if cfg, ok := wire(provider("kimi")).modelConfig(config.Model{ID: "m", Temperature: &temp}).(*kimi.ChatConfig); !ok {
		t.Fatalf("kimi config = %T, want *kimi.ChatConfig", cfg)
	} else if cfg == nil {
		t.Fatal("kimi config is nil")
	}

	if cfg, ok := wire(provider("openai")).modelConfig(config.Model{ID: "m", Temperature: &temp}).(*openaiGo.ChatCompletionNewParams); !ok {
		t.Fatalf("openai config = %T, want *openaiGo.ChatCompletionNewParams", cfg)
	} else if !cfg.Temperature.Valid() || cfg.Temperature.Value != temp {
		t.Errorf("temperature = %v, want %v", cfg.Temperature.Value, temp)
	}

	unknown := config.Provider{Name: "myproxy", Model: []config.Model{{ID: "m"}}}
	if cfg, ok := wire(unknown).modelConfig(config.Model{ID: "m"}).(*openaiGo.ChatCompletionNewParams); !ok {
		t.Fatalf("generic config = %T, want *openaiGo.ChatCompletionNewParams", cfg)
	} else if cfg == nil {
		t.Fatal("generic config is nil")
	}
}
