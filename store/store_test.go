package store

import (
	"encoding/json/v2"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

func transcript() []*ai.Message {
	return []*ai.Message{
		ai.NewUserTextMessage("hi"),
		{
			Role: ai.RoleModel,
			Content: []*ai.Part{
				ai.NewReasoningPart("thinking", nil),
				ai.NewTextPart("hello"),
			},
			Metadata: map[string]any{"messageId": "msg_1"},
		},
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	ws, err := Open("/work/project")
	if err != nil {
		t.Fatalf("open: %v", err)
	}

	meta, err := ws.Save("sess_abc", SessionMeta{Model: "openai/gpt-4o"}, transcript())
	if err != nil {
		t.Fatalf("save: %v", err)
	}
	if meta.CreatedAt.IsZero() || meta.UpdatedAt.IsZero() {
		t.Fatalf("timestamps not set: %+v", meta)
	}

	got, loaded, err := ws.Load("sess_abc")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if loaded.Model != "openai/gpt-4o" || !loaded.CreatedAt.Equal(meta.CreatedAt) {
		t.Errorf("meta = %+v, want model and createdAt preserved from %+v", loaded, meta)
	}
	if len(got) != 2 {
		t.Fatalf("messages = %d, want 2", len(got))
	}
	if got[0].Role != ai.RoleUser || got[0].Text() != "hi" {
		t.Errorf("first message = %+v, want the user's question", got[0])
	}
	if id, _ := got[1].Metadata["messageId"].(string); id != "msg_1" {
		t.Errorf("model messageId = %q, want msg_1", id)
	}
	if p := got[1].Content[0]; p.Kind != ai.PartReasoning || p.Text != "thinking" {
		t.Errorf("reasoning part = %+v, want it stored verbatim", p)
	}
	if p := got[1].Content[1]; p.Kind != ai.PartText || p.Text != "hello" {
		t.Errorf("text part = %+v, want hello", p)
	}

	// A second save must keep the original creation time.
	meta2, err := ws.Save("sess_abc", SessionMeta{Model: "openai/gpt-4o", CreatedAt: loaded.CreatedAt}, transcript())
	if err != nil {
		t.Fatalf("second save: %v", err)
	}
	if !meta2.CreatedAt.Equal(meta.CreatedAt) {
		t.Errorf("createdAt drifted: %v then %v", meta.CreatedAt, meta2.CreatedAt)
	}

	// Conversation files are private: they can hold code and secrets.
	path := filepath.Join(ws.dir, "sessions", "sess_abc.json")
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("file mode = %o, want 600", perm)
	}
	if _, err := os.Stat(filepath.Join(ws.dir, "workspace.json")); err != nil {
		t.Errorf("workspace metadata: %v", err)
	}
}

func TestLoadQuarantinesBadFiles(t *testing.T) {
	tests := []struct {
		name  string
		write func(t *testing.T, path string)
	}{
		{
			name: "corrupt json",
			write: func(t *testing.T, path string) {
				if err := os.WriteFile(path, []byte("{not json"), 0o600); err != nil {
					t.Fatal(err)
				}
			},
		},
		{
			name: "foreign sessionId",
			write: func(t *testing.T, path string) {
				writeEnvelope(t, path, map[string]any{
					"version": 1, "sessionId": "sess_other", "messages": []any{},
				})
			},
		},
		{
			name: "unknown version",
			write: func(t *testing.T, path string) {
				writeEnvelope(t, path, map[string]any{
					"version": 99, "sessionId": "sess_x", "messages": []any{},
				})
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("HOME", t.TempDir())
			ws, err := Open("/work/project")
			if err != nil {
				t.Fatalf("open: %v", err)
			}
			path, err := ws.sessionPath("sess_x")
			if err != nil {
				t.Fatalf("session path: %v", err)
			}
			if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
				t.Fatal(err)
			}
			tt.write(t, path)

			if _, _, err := ws.Load("sess_x"); err == nil || errors.Is(err, fs.ErrNotExist) {
				t.Fatalf("load error = %v, want a quarantine error", err)
			}
			if _, err := os.Stat(path + ".bak"); err != nil {
				t.Errorf("original not quarantined: %v", err)
			}
			// After quarantine the session reads as simply absent.
			if _, _, err := ws.Load("sess_x"); !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("load after quarantine = %v, want ErrNotExist", err)
			}
		})
	}
}

func writeEnvelope(t *testing.T, path string, fields map[string]any) {
	t.Helper()
	data, err := json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestWorkspacesArePerCwd(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	one, err := Open("/work/one")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	again, err := Open("/work/one")
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if one.dir != again.dir {
		t.Errorf("same cwd resolved differently: %q vs %q", one.dir, again.dir)
	}
	if _, err := one.Save("sess_shared", SessionMeta{}, transcript()); err != nil {
		t.Fatalf("save: %v", err)
	}

	two, err := Open("/work/two")
	if err != nil {
		t.Fatalf("open two: %v", err)
	}
	if two.dir == one.dir {
		t.Fatal("different cwds share a workspace")
	}
	if _, _, err := two.Load("sess_shared"); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("load in the other workspace = %v, want ErrNotExist", err)
	}
}

func TestSaveRejectsUnsafeSessionID(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	ws, err := Open("/work/project")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	// The id reaches the agent from the client on resume.
	for _, id := range []string{"", "../evil", "a/b", `a\b`, "a b", ".hidden"} {
		if _, err := ws.Save(id, SessionMeta{}, nil); err == nil {
			t.Errorf("Save(%q) succeeded, want a rejection", id)
		}
		if _, _, err := ws.Load(id); err == nil {
			t.Errorf("Load(%q) succeeded, want a rejection", id)
		}
	}
}
