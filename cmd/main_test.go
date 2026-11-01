package main

import (
	"strings"
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

// provider builds a provider entry whose name and provider value match, the
// simple shape of a config naming a dedicated plugin.
func provider(plugin string) config.Provider {
	return config.Provider{Name: plugin, Provider: plugin, APIKey: "k", Model: []config.Model{{ID: "m"}}}
}

// wiredFor wires p and fails the test if it does not wire.
func wiredFor(t *testing.T, p config.Provider) wired {
	t.Helper()
	w, err := wire(p)
	if err != nil {
		t.Fatalf("wire(%q): %v", p.Provider, err)
	}
	return w
}

func TestWireDedicatedPlugins(t *testing.T) {
	clearKeyEnvs(t)
	for _, name := range []string{"openai", "anthropic", "dashscope", "deepseek", "kimi", "openrouter", "xai", "zai"} {
		if _, ok := wiredFor(t, provider(name)).plugin.(*compat_oai.OpenAICompatible); ok {
			t.Errorf("wire(%q) served by the generic plugin, want the dedicated one", name)
		}
	}
}

// TestWireGeneric checks the explicit openai-compatible value: it needs no
// key, and the plugin registers models under the entry's own name.
func TestWireGeneric(t *testing.T) {
	clearKeyEnvs(t)
	p := config.Provider{Name: "myproxy", Provider: "openai-compatible", Model: []config.Model{{ID: "m"}}}
	w := wiredFor(t, p)
	plugin, ok := w.plugin.(*compat_oai.OpenAICompatible)
	if !ok {
		t.Fatalf("plugin = %T, want *compat_oai.OpenAICompatible", w.plugin)
	}
	if plugin.Provider != "myproxy" {
		t.Errorf("plugin provider = %q, want the entry name %q", plugin.Provider, "myproxy")
	}
}

// TestWireUnknownProvider checks that an unknown provider value is an error
// rather than a silent fallthrough to the generic plugin.
func TestWireUnknownProvider(t *testing.T) {
	clearKeyEnvs(t)
	p := config.Provider{Name: "myproxy", Provider: "myproxy", Model: []config.Model{{ID: "m"}}}
	if _, err := wire(p); err == nil || !strings.Contains(err.Error(), `unknown provider "myproxy"`) {
		t.Fatalf("err = %v, want unknown provider", err)
	}
}

// TestWireDedicatedWithoutKey checks that a dedicated plugin with no key
// anywhere is an error rather than a fallthrough to the generic plugin.
func TestWireDedicatedWithoutKey(t *testing.T) {
	clearKeyEnvs(t)
	p := config.Provider{Name: "ds", Provider: "deepseek", Model: []config.Model{{ID: "m"}}}
	if _, err := wire(p); err == nil || !strings.Contains(err.Error(), "no API key") {
		t.Fatalf("err = %v, want a missing key error", err)
	}
}

func TestWireKeyFromEnvironment(t *testing.T) {
	clearKeyEnvs(t)
	t.Setenv("DEEPSEEK_API_KEY", "from-env")
	if _, ok := wiredFor(t, provider("deepseek")).plugin.(*deepseek.DeepSeek); !ok {
		t.Fatal("deepseek with a key only in the environment is not served by its plugin")
	}
}

// TestWireAllRefusesDuplicatePlugin checks that two entries naming the same
// dedicated plugin are refused: they would collide on the plugin's fixed
// registration name rather than add a second provider.
func TestWireAllRefusesDuplicatePlugin(t *testing.T) {
	clearKeyEnvs(t)
	cfg := &config.Config{Provider: []config.Provider{
		provider("deepseek"),
		{Name: "ds-proxy", Provider: "deepseek", APIKey: "k", Model: []config.Model{{ID: "m"}}},
	}}
	if _, err := wireAll(cfg); err == nil || !strings.Contains(err.Error(), "both register") {
		t.Fatalf("err = %v, want a duplicate plugin error", err)
	}
}

func TestModelConfig(t *testing.T) {
	clearKeyEnvs(t)
	temp := 0.7

	if cfg, ok := wiredFor(t, provider("deepseek")).modelConfig(config.Model{ID: "m", Temperature: &temp}).(*deepseek.ChatConfig); !ok {
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
	if cfg, ok := wiredFor(t, provider("kimi")).modelConfig(config.Model{ID: "m", Temperature: &temp}).(*kimi.ChatConfig); !ok {
		t.Fatalf("kimi config = %T, want *kimi.ChatConfig", cfg)
	} else if cfg == nil {
		t.Fatal("kimi config is nil")
	}

	if cfg, ok := wiredFor(t, provider("openai")).modelConfig(config.Model{ID: "m", Temperature: &temp}).(*openaiGo.ChatCompletionNewParams); !ok {
		t.Fatalf("openai config = %T, want *openaiGo.ChatCompletionNewParams", cfg)
	} else if !cfg.Temperature.Valid() || cfg.Temperature.Value != temp {
		t.Errorf("temperature = %v, want %v", cfg.Temperature.Value, temp)
	}

	generic := config.Provider{Name: "myproxy", Provider: "openai-compatible", Model: []config.Model{{ID: "m"}}}
	if cfg, ok := wiredFor(t, generic).modelConfig(config.Model{ID: "m"}).(*openaiGo.ChatCompletionNewParams); !ok {
		t.Fatalf("generic config = %T, want *openaiGo.ChatCompletionNewParams", cfg)
	} else if cfg == nil {
		t.Fatal("generic config is nil")
	}
}
