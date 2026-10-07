package agent

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"genkit-code-agent/store"

	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/deepseek"
)

// fakeDeepSeek is an OpenAI-compatible endpoint that records every request it
// serves and answers every turn with one canned streaming reply.
type fakeDeepSeek struct {
	mu     sync.Mutex
	bodies []string
}

func startFakeDeepSeek(t *testing.T) *fakeDeepSeek {
	t.Helper()
	f := &fakeDeepSeek{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		f.mu.Lock()
		f.bodies = append(f.bodies, string(body))
		f.mu.Unlock()

		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"object":"list","data":[]}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: %s\n\n",
			`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{"role":"assistant","content":"the answer"},"finish_reason":null}]}`)
		if flusher != nil {
			flusher.Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)
	t.Setenv("DEEPSEEK_BASE_URL", srv.URL)
	return f
}

func (f *fakeDeepSeek) requests() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.bodies...)
}

func fakeModel() Model {
	return Model{Ref: "deepseek/deepseek-flash", Name: "deepseek-flash", Group: "deepseek"}
}

// TestTranscriptSurvivesRestart runs two turns, throws the agent away as if
// the process had exited, resumes the same session id on the same cwd with a
// fresh agent, and checks the third turn's request still carries the first
// two exchanges: persistence, not just in-memory history.
func TestTranscriptSurvivesRestart(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	fake := startFakeDeepSeek(t)
	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))
	cwd := t.TempDir()

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	_, client1, conn1, _ := openAgentPipes(t, g, fakeModel())
	sess, err := client1.NewSession(ctx, acp.NewSessionRequest{Cwd: cwd, McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	for _, q := range []string{"what is the first question", "what is the second question"} {
		if _, err := client1.Prompt(ctx, acp.PromptRequest{
			SessionId: sess.SessionId,
			Prompt:    []acp.ContentBlock{acp.TextBlock(q)},
		}); err != nil {
			t.Fatalf("prompt %q: %v", q, err)
		}
	}
	_ = conn1.Done()
	_ = client1.Done()

	// A fresh agent knows nothing but the cwd and the session id.
	ag2, client2, conn2, _ := openAgentPipes(t, g, fakeModel())
	if _, err := client2.ResumeSession(ctx, acp.ResumeSessionRequest{
		SessionId: sess.SessionId,
		Cwd:       cwd,
	}); err != nil {
		t.Fatalf("resume: %v", err)
	}
	s2, ok := ag2.get(string(sess.SessionId))
	if !ok {
		t.Fatal("resumed session not registered")
	}
	stored, _, err := s2.load(string(sess.SessionId))
	if err != nil {
		t.Fatalf("read stored transcript: %v", err)
	}
	if len(stored) != 4 {
		t.Fatalf("stored transcript = %d messages, want 4", len(stored))
	}

	if _, err := client2.Prompt(ctx, acp.PromptRequest{
		SessionId: sess.SessionId,
		Prompt:    []acp.ContentBlock{acp.TextBlock("what is the third question")},
	}); err != nil {
		t.Fatalf("third prompt: %v", err)
	}
	_ = conn2.Done()
	_ = client2.Done()

	reqs := fake.requests()
	if len(reqs) != 3 {
		t.Fatalf("model requests = %d, want 3", len(reqs))
	}
	last := reqs[2]
	// The restored exchange must reach the model ahead of the new question,
	// in conversation order.
	at := func(s string) int {
		i := strings.Index(last, s)
		if i < 0 {
			t.Fatalf("restarted request missing %q:\n%s", s, last)
		}
		return i
	}
	q1, ans1, q2, ans2, q3 := at("what is the first question"), at("the answer"),
		at("what is the second question"), strings.LastIndex(last, "the answer"),
		at("what is the third question")
	if !(q1 < ans1 && ans1 < q2 && q2 < ans2 && ans2 < q3) {
		t.Errorf("history order = %d,%d,%d,%d,%d, want first question, answer, second question, answer, third question",
			q1, ans1, q2, ans2, q3)
	}
}

// TestLoadSessionReplaysTranscript checks the other half of persistence:
// session/load streams the stored conversation back as notifications, under
// the message ids the turns were saved with.
func TestLoadSessionReplaysTranscript(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	startFakeDeepSeek(t)
	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))
	cwd := t.TempDir()

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	givenID := "11111111-2222-3333-4444-555555555555"
	_, client1, conn1, _ := openAgentPipes(t, g, fakeModel())
	sess, err := client1.NewSession(ctx, acp.NewSessionRequest{Cwd: cwd, McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	if _, err := client1.Prompt(ctx, acp.PromptRequest{
		SessionId: sess.SessionId,
		MessageId: &givenID,
		Prompt:    []acp.ContentBlock{acp.TextBlock("what is the first question")},
	}); err != nil {
		t.Fatalf("first prompt: %v", err)
	}
	if _, err := client1.Prompt(ctx, acp.PromptRequest{
		SessionId: sess.SessionId,
		Prompt:    []acp.ContentBlock{acp.TextBlock("what is the second question")},
	}); err != nil {
		t.Fatalf("second prompt: %v", err)
	}
	_ = conn1.Done()
	_ = client1.Done()

	_, client2, conn2, rc := openAgentPipes(t, g, fakeModel())
	resp, err := client2.LoadSession(ctx, acp.LoadSessionRequest{
		SessionId:  sess.SessionId,
		Cwd:        cwd,
		McpServers: []acp.McpServer{},
	})
	if err != nil {
		t.Fatalf("load session: %v", err)
	}
	if len(resp.ConfigOptions) != 1 {
		t.Errorf("config options = %+v, want the model selector back", resp.ConfigOptions)
	}
	_ = conn2.Done()
	_ = client2.Done()

	rc.testClient.mu.Lock()
	events := append([]streamEvent(nil), rc.events...)
	rc.testClient.mu.Unlock()

	want := []struct {
		channel string
		text    string
	}{
		{"user", "what is the first question"},
		{"message", "the answer"},
		{"user", "what is the second question"},
		{"message", "the answer"},
	}
	if len(events) != len(want) {
		t.Fatalf("replay events = %+v, want %d", events, len(want))
	}
	for i, w := range want {
		if events[i].Channel != w.channel || events[i].Text != w.text {
			t.Errorf("event %d = %+v, want {%s %q}", i, events[i], w.channel, w.text)
		}
	}
	// Ids regroup the replay: the client's id on its own message, a fresh
	// UUID on the one the agent assigned, msg_ ids on the answers, one per
	// turn.
	if events[0].MessageID != givenID {
		t.Errorf("first user event id = %q, want the client's %q", events[0].MessageID, givenID)
	}
	if events[2].MessageID == "" || events[2].MessageID == givenID || strings.HasPrefix(events[2].MessageID, "msg_") {
		t.Errorf("second user event id = %q, want a distinct plain UUID", events[2].MessageID)
	}
	if !strings.HasPrefix(events[1].MessageID, "msg_") || events[3].MessageID == events[1].MessageID {
		t.Errorf("answer ids = %q, %q, want distinct msg_ ids per turn", events[1].MessageID, events[3].MessageID)
	}
}

// TestResumeRestoresTranscriptAndModel exercises the resume path directly:
// resuming a stored id points the session back at its workspace, so the
// stored conversation reads back through it, and the model it last ran on
// comes back with it even after a selection moved off the default.
func TestResumeRestoresTranscriptAndModel(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cwd := t.TempDir()

	ws, err := store.Open(cwd)
	if err != nil {
		t.Fatalf("open workspace: %v", err)
	}
	stored := []*ai.Message{
		ai.NewUserTextMessage("hi"),
		{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("yo")}},
	}
	if _, err := ws.Save("sess_restore", store.SessionMeta{Model: "openai/gpt-4o"}, stored); err != nil {
		t.Fatalf("seed store: %v", err)
	}

	ag := New(nil, testModels())
	if _, err := ag.ResumeSession(context.Background(), acp.ResumeSessionRequest{
		SessionId: "sess_restore",
		Cwd:       cwd,
	}); err != nil {
		t.Fatalf("resume: %v", err)
	}

	s, ok := ag.get("sess_restore")
	if !ok {
		t.Fatal("resumed session not registered")
	}
	loaded, _, err := s.load("sess_restore")
	if err != nil {
		t.Fatalf("read stored transcript: %v", err)
	}
	if len(loaded) != 2 || loaded[0].Text() != "hi" {
		t.Errorf("stored transcript = %+v, want the stored exchange", loaded)
	}
	if model := s.currentModel(); model != "openai/gpt-4o" {
		t.Errorf("model = %q, want the stored openai/gpt-4o", model)
	}
}
