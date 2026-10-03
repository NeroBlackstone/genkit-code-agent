package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func write(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "gca.json")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

const valid = `{
  "provider": [
    {
      "name": "deepseek",
      "baseUrl": "https://api.deepseek.com",
      "apiKey": "k",
      "model": [
        {"id": "deepseek-flash", "temperature": 0.7, "maxTokens": 1000000},
        {"id": "deepseek-chat"}
      ]
    },
    {"name": "openai", "baseUrl": "https://api.openai.com/v1", "model": [{"id": "gpt-4o-mini"}]}
  ]
}`

func TestLoadMultiProvider(t *testing.T) {
	cfg, err := Load(write(t, valid))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(cfg.Provider) != 2 {
		t.Fatalf("providers = %d, want 2", len(cfg.Provider))
	}
	p, m := cfg.Active()
	if p.Name != "deepseek" || m.ID != "deepseek-flash" {
		t.Fatalf("Active = %s/%s, want deepseek/deepseek-flash", p.Name, m.ID)
	}
	if len(cfg.Provider[0].Model) != 2 || cfg.Provider[0].Model[1].ID != "deepseek-chat" {
		t.Fatalf("first provider models = %+v", cfg.Provider[0].Model)
	}
	if m.Temperature == nil || *m.Temperature != 0.7 || m.MaxTokens == nil || *m.MaxTokens != 1000000 {
		t.Fatalf("model params = %+v", m)
	}
}

func TestLoadErrors(t *testing.T) {
	cases := []struct {
		name, body, want string
	}{
		{"empty", `{"provider": []}`, "at least one provider"},
		{"missing name", `{"provider": [{"model": [{"id": "m"}]}]}`, "provider[0].name"},
		{"duplicate name", `{"provider": [{"name": "a", "model": [{"id": "m"}]}, {"name": "a", "model": [{"id": "m"}]}]}`, "duplicate provider"},
		{"no models", `{"provider": [{"name": "a"}]}`, "at least one model"},
		{"missing model id", `{"provider": [{"name": "a", "model": [{}]}]}`, "model[0].id"},
		{"bad json", `{`, "parse config"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Load(write(t, tc.body))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want containing %q", err, tc.want)
			}
		})
	}
}
