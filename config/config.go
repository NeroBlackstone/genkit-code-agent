package config

import (
	"encoding/json/v2"
	"fmt"
	"os"
	"path/filepath"
)

type Model struct {
	ID          string   `json:"id"`
	Temperature *float64 `json:"temperature,omitempty"`
}

type Provider struct {
	Name    string  `json:"name"`
	BaseURL string  `json:"baseUrl"`
	APIKey  string  `json:"apiKey"`
	Model   []Model `json:"model"`
}

type Config struct {
	Provider []Provider `json:"provider"`
}

// Active returns the provider and model to run: the first model of the first
// provider.
func (c *Config) Active() (Provider, Model) {
	p := c.Provider[0]
	return p, p.Model[0]
}

// DefaultPath returns ~/.gca/gca.json.
func DefaultPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolve home dir: %w", err)
	}
	return filepath.Join(home, ".gca", "gca.json"), nil
}

// Load reads the config at path.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config %s: %w", path, err)
	}
	if len(cfg.Provider) == 0 {
		return nil, fmt.Errorf("config %s: at least one provider is required", path)
	}
	seen := make(map[string]bool, len(cfg.Provider))
	for i, p := range cfg.Provider {
		if p.Name == "" {
			return nil, fmt.Errorf("config %s: provider[%d].name is required", path, i)
		}
		if seen[p.Name] {
			return nil, fmt.Errorf("config %s: duplicate provider %q", path, p.Name)
		}
		seen[p.Name] = true
		if len(p.Model) == 0 {
			return nil, fmt.Errorf("config %s: provider %q: at least one model is required", path, p.Name)
		}
		for j, m := range p.Model {
			if m.ID == "" {
				return nil, fmt.Errorf("config %s: provider %q model[%d].id is required", path, p.Name, j)
			}
		}
	}
	return &cfg, nil
}
