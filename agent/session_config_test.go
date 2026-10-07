package agent

import (
	"context"
	"testing"

	"github.com/coder/acp-go-sdk"
)

func testModels() []Model {
	return []Model{
		{Ref: "deepseek/deepseek-flash", Name: "deepseek-flash", Group: "deepseek"},
		{Ref: "deepseek/deepseek-v4-pro", Name: "deepseek-v4-pro", Group: "deepseek"},
		{Ref: "openai/gpt-4o", Name: "gpt-4o", Group: "openai"},
	}
}

func modelOption(t *testing.T, options []acp.SessionConfigOption) *acp.SessionConfigOptionSelect {
	t.Helper()
	if len(options) != 1 {
		t.Fatalf("options = %d, want 1", len(options))
	}
	sel := options[0].Select
	if sel == nil {
		t.Fatalf("option = %+v, want a select", options[0])
	}
	return sel
}

func TestNewSessionOffersModelOption(t *testing.T) {
	t.Setenv("HOME", t.TempDir()) // keep the session store out of the real ~/.gca
	ag := New(nil, testModels())

	resp, err := ag.NewSession(context.Background(), acp.NewSessionRequest{Cwd: "/tmp", McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}

	sel := modelOption(t, resp.ConfigOptions)
	if sel.Id != modelConfigOptionID {
		t.Errorf("id = %q, want %q", sel.Id, modelConfigOptionID)
	}
	if sel.Category == nil || *sel.Category != acp.SessionConfigOptionCategoryModel {
		t.Errorf("category = %v, want %q", sel.Category, acp.SessionConfigOptionCategoryModel)
	}
	// The spec requires a default: the first configured model.
	if sel.CurrentValue != "deepseek/deepseek-flash" {
		t.Errorf("current value = %q, want the first configured model", sel.CurrentValue)
	}

	if sel.Options.Grouped == nil {
		t.Fatalf("options = %+v, want grouped by provider", sel.Options)
	}
	if len(*sel.Options.Grouped) != 2 {
		t.Fatalf("groups = %d, want 2", len(*sel.Options.Grouped))
	}
	first := (*sel.Options.Grouped)[0]
	if first.Group != "deepseek" || len(first.Options) != 2 {
		t.Errorf("first group = %+v, want deepseek with 2 models", first)
	}
	if second := (*sel.Options.Grouped)[1]; second.Group != "openai" || len(second.Options) != 1 {
		t.Errorf("second group = %+v, want openai with 1 model", second)
	}
}

func TestNewSessionWithoutModels(t *testing.T) {
	t.Setenv("HOME", t.TempDir()) // keep the session store out of the real ~/.gca
	resp, err := New(nil, nil).NewSession(context.Background(), acp.NewSessionRequest{Cwd: "/tmp", McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	if len(resp.ConfigOptions) != 0 {
		t.Fatalf("options = %+v, want none", resp.ConfigOptions)
	}
}

func TestSetSessionConfigOption(t *testing.T) {
	t.Setenv("HOME", t.TempDir()) // keep the session store out of the real ~/.gca
	ag := New(nil, testModels())
	resp, err := ag.NewSession(context.Background(), acp.NewSessionRequest{Cwd: "/tmp", McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}

	set := func(configID acp.SessionConfigId, value string) (acp.SetSessionConfigOptionResponse, error) {
		return ag.SetSessionConfigOption(context.Background(), acp.SetSessionConfigOptionRequest{
			ValueId: &acp.SetSessionConfigOptionValueId{
				SessionId: resp.SessionId,
				ConfigId:  configID,
				Value:     acp.SessionConfigValueId(value),
			},
		})
	}

	t.Run("selects a model", func(t *testing.T) {
		out, err := set(modelConfigOptionID, "openai/gpt-4o")
		if err != nil {
			t.Fatalf("set: %v", err)
		}
		// The response is the complete state, not the change.
		sel := modelOption(t, out.ConfigOptions)
		if sel.CurrentValue != "openai/gpt-4o" {
			t.Errorf("current value = %q, want openai/gpt-4o", sel.CurrentValue)
		}
		s, ok := ag.get(string(resp.SessionId))
		if !ok {
			t.Fatal("session not registered")
		}
		if got := s.currentModel(); got != "openai/gpt-4o" {
			t.Errorf("session model = %q, want openai/gpt-4o", got)
		}
	})

	t.Run("rejects an unknown value", func(t *testing.T) {
		if _, err := set(modelConfigOptionID, "deepseek/nope"); err == nil {
			t.Fatal("set an unlisted model, want an error")
		}
	})

	t.Run("rejects an unknown option", func(t *testing.T) {
		if _, err := set("mode", "code"); err == nil {
			t.Fatal("set an unknown config option, want an error")
		}
	})

	t.Run("rejects a non-select", func(t *testing.T) {
		if _, err := ag.SetSessionConfigOption(context.Background(), acp.SetSessionConfigOptionRequest{}); err == nil {
			t.Fatal("set without a value id, want an error")
		}
	})
}

func TestResumeSessionAdoptsClientID(t *testing.T) {
	t.Setenv("HOME", t.TempDir()) // keep the session store out of the real ~/.gca
	ag := New(nil, testModels())

	out, err := ag.ResumeSession(context.Background(), acp.ResumeSessionRequest{
		SessionId: "sess_from_before_restart",
		Cwd:       "/tmp",
	})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	// Nothing to replay: the response carries the config state and no history.
	sel := modelOption(t, out.ConfigOptions)
	if sel.CurrentValue != "deepseek/deepseek-flash" {
		t.Errorf("current value = %q, want the first configured model", sel.CurrentValue)
	}
	if _, ok := ag.sessions["sess_from_before_restart"]; !ok {
		t.Fatal("resumed id not registered")
	}
}

func TestResumeSessionKeepsModelSelection(t *testing.T) {
	t.Setenv("HOME", t.TempDir()) // keep the session store out of the real ~/.gca
	ag := New(nil, testModels())
	resp, err := ag.NewSession(context.Background(), acp.NewSessionRequest{Cwd: "/tmp", McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	if _, err := ag.SetSessionConfigOption(context.Background(), acp.SetSessionConfigOptionRequest{
		ValueId: &acp.SetSessionConfigOptionValueId{
			SessionId: resp.SessionId,
			ConfigId:  modelConfigOptionID,
			Value:     "openai/gpt-4o",
		},
	}); err != nil {
		t.Fatalf("set: %v", err)
	}

	out, err := ag.ResumeSession(context.Background(), acp.ResumeSessionRequest{SessionId: resp.SessionId, Cwd: "/tmp"})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	sel := modelOption(t, out.ConfigOptions)
	if sel.CurrentValue != "openai/gpt-4o" {
		t.Errorf("current value = %q, want the selection to survive a resume", sel.CurrentValue)
	}
}
