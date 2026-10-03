package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/deepseek"
)

// streamEvent is one session/update as the client sees it: the ACP channel it
// arrived on, the messageId it groups under, and the text it carried.
type streamEvent struct {
	Channel   string
	MessageID string
	Text      string
}

// recordingClient is the testClient with SessionUpdate overridden to record
// both ACP channels in arrival order, which is what the split is judged on.
type recordingClient struct {
	testClient
	events []streamEvent
}

func (c *recordingClient) SessionUpdate(_ context.Context, p acp.SessionNotification) error {
	c.testClient.mu.Lock()
	defer c.testClient.mu.Unlock()
	switch {
	case p.Update.UserMessageChunk != nil && p.Update.UserMessageChunk.Content.Text != nil:
		id := ""
		if p.Update.UserMessageChunk.MessageId != nil {
			id = *p.Update.UserMessageChunk.MessageId
		}
		c.events = append(c.events, streamEvent{"user", id, p.Update.UserMessageChunk.Content.Text.Text})
	case p.Update.AgentMessageChunk != nil && p.Update.AgentMessageChunk.Content.Text != nil:
		id := ""
		if p.Update.AgentMessageChunk.MessageId != nil {
			id = *p.Update.AgentMessageChunk.MessageId
		}
		c.events = append(c.events, streamEvent{"message", id, p.Update.AgentMessageChunk.Content.Text.Text})
	case p.Update.AgentThoughtChunk != nil && p.Update.AgentThoughtChunk.Content.Text != nil:
		id := ""
		if p.Update.AgentThoughtChunk.MessageId != nil {
			id = *p.Update.AgentThoughtChunk.MessageId
		}
		c.events = append(c.events, streamEvent{"thought", id, p.Update.AgentThoughtChunk.Content.Text.Text})
	}
	return nil
}

// openAgentPipes wires a real agent and client over in-memory pipes for g and
// model, returning the agent, the client connection to drive, and its
// recording client.
func openAgentPipes(t *testing.T, g *genkit.Genkit, model Model) (*Agent, *acp.ClientSideConnection, *acp.AgentSideConnection, *recordingClient) {
	t.Helper()

	agentToClientR, agentToClientW := io.Pipe()
	clientToAgentR, clientToAgentW := io.Pipe()

	ag := New()
	agentConn := acp.NewAgentSideConnection(ag, agentToClientW, clientToAgentR)
	ag.SetAgentConnection(agentConn)
	ag.SetModels(g, []Model{model})

	rc := &recordingClient{}
	clientConn := acp.NewClientSideConnection(rc, clientToAgentW, agentToClientR)

	if _, err := clientConn.Initialize(context.Background(), acp.InitializeRequest{
		ProtocolVersion: acp.ProtocolVersionNumber,
		ClientInfo:      &acp.Implementation{Name: "test-client", Version: "0.0.1"},
	}); err != nil {
		t.Fatalf("initialize: %v", err)
	}
	return ag, clientConn, agentConn, rc
}

// runTurnEvents drives one prompt through the real ACP pipes and returns the
// updates the client received, in order.
func runTurnEvents(t *testing.T, g *genkit.Genkit, model Model, prompt string) []streamEvent {
	t.Helper()

	_, clientConn, agentConn, rc := openAgentPipes(t, g, model)

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	sess, err := clientConn.NewSession(ctx, acp.NewSessionRequest{Cwd: "/tmp", McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	if _, err := clientConn.Prompt(ctx, acp.PromptRequest{
		SessionId: sess.SessionId,
		Prompt:    []acp.ContentBlock{acp.TextBlock(prompt)},
	}); err != nil {
		t.Fatalf("prompt: %v", err)
	}

	_ = agentConn.Done()
	_ = clientConn.Done()

	rc.testClient.mu.Lock()
	defer rc.testClient.mu.Unlock()
	return rc.events
}

// checkMessageIDs asserts every update carries a messageId and that one id
// groups the whole turn, thinking and answer alike, the way Kilo's output
// labels them.
func checkMessageIDs(t *testing.T, events []streamEvent) {
	t.Helper()
	seen := ""
	for i, e := range events {
		if e.MessageID == "" {
			t.Errorf("event %d (%s) has no messageId: %+v", i, e.Channel, e)
			continue
		}
		if seen == "" {
			seen = e.MessageID
			if !strings.HasPrefix(seen, "msg_") {
				t.Errorf("messageId = %q, want Kilo's msg_ prefix", seen)
			}
		} else if seen != e.MessageID {
			t.Errorf("messageId changed within one turn: %q then %q (event %d)", seen, e.MessageID, i)
		}
	}
}

func joined(events []streamEvent, channel string) string {
	var sb strings.Builder
	for _, e := range events {
		if e.Channel == channel {
			sb.WriteString(e.Text)
		}
	}
	return sb.String()
}

// firstIndexOf returns the position of the first event on channel, or -1.
func firstIndexOf(events []streamEvent, channel string) int {
	for i, e := range events {
		if e.Channel == channel {
			return i
		}
	}
	return -1
}

// TestRunTurnSplitsDeepSeekStream replays a DeepSeek-shaped stream through the
// whole path: SSE endpoint → genkit deepseek plugin → genkit chunk → ACP
// update. reasoning_content deltas must land on the thought channel and content
// deltas on the message channel, with no interleaving after the answer starts.
func TestRunTurnSplitsDeepSeekStream(t *testing.T) {
	t.Setenv("HOME", t.TempDir()) // keep the session store out of the real ~/.gca
	var (
		mu        sync.Mutex
		reqBodies []string
	)

	chunks := []string{
		`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"91 = 7 x 13，"},"finish_reason":null}]}`,
		`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{"reasoning_content":"是合数，回答"},"finish_reason":null}]}`,
		`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{"content":"不是质数，"},"finish_reason":null}]}`,
		`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{"content":"91 = 7 x 13。"},"finish_reason":null}]}`,
		`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}`,
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		reqBodies = append(reqBodies, string(body))
		mu.Unlock()

		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			// Dynamic model listing asks the endpoint what it serves; answer
			// with an empty catalog rather than failing the lookup.
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"object":"list","data":[]}`)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			if flusher != nil {
				flusher.Flush()
			}
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	t.Setenv("DEEPSEEK_BASE_URL", srv.URL)

	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))

	events := runTurnEvents(t, g, Model{
		Ref:   "deepseek/deepseek-flash",
		Name:  "deepseek-flash",
		Group: "deepseek",
		Config: &deepseek.ChatConfig{
			Temperature: nil,
			Thinking:    &deepseek.ThinkingConfig{Type: deepseek.ThinkingTypeEnabled},
		},
	}, "91是质数吗？")

	if got, want := joined(events, "thought"), "91 = 7 x 13，是合数，回答"; got != want {
		t.Errorf("thought channel = %q, want %q", got, want)
	}
	if got, want := joined(events, "message"), "不是质数，91 = 7 x 13。"; got != want {
		t.Errorf("message channel = %q, want %q", got, want)
	}
	thoughtAt, messageAt := firstIndexOf(events, "thought"), firstIndexOf(events, "message")
	if thoughtAt < 0 || messageAt < 0 {
		t.Fatalf("channels seen: thought=%d message=%d, events=%+v", thoughtAt, messageAt, events)
	}
	for i, e := range events {
		if e.Channel == "thought" && i > messageAt {
			t.Errorf("thought chunk after the answer started: event %d = %+v", i, e)
		}
	}
	checkMessageIDs(t, events)

	mu.Lock()
	defer mu.Unlock()
	if len(reqBodies) == 0 {
		t.Fatal("no chat completion request reached the endpoint")
	}
	var req map[string]any
	if err := json.Unmarshal([]byte(reqBodies[0]), &req); err != nil {
		t.Fatalf("request body: %v", err)
	}
	if req["model"] != "deepseek-flash" {
		t.Errorf("model = %v, want deepseek-flash", req["model"])
	}
	if req["stream"] != true {
		t.Errorf("stream = %v, want true", req["stream"])
	}
	thinking, _ := req["thinking"].(map[string]any)
	if thinking["type"] != "enabled" {
		t.Errorf("thinking = %v, want type enabled", req["thinking"])
	}
}

// TestRunTurnSplitsDeepSeekStreamLive is the same split checked against the
// real DeepSeek API. It runs only with DEEPSEEK_API_KEY set, and prints the
// channel sequence so the routing can be read directly.
func TestRunTurnSplitsDeepSeekStreamLive(t *testing.T) {
	t.Setenv("HOME", t.TempDir()) // keep the session store out of the real ~/.gca
	if os.Getenv("DEEPSEEK_API_KEY") == "" {
		t.Skip("DEEPSEEK_API_KEY is not set")
	}
	modelID := os.Getenv("GCA_TEST_MODEL")
	if modelID == "" {
		modelID = "deepseek-flash"
	}

	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{}))

	events := runTurnEvents(t, g, Model{
		Ref:    "deepseek/" + modelID,
		Name:   modelID,
		Group:  "deepseek",
		Config: &deepseek.ChatConfig{Thinking: &deepseek.ThinkingConfig{Type: deepseek.ThinkingTypeEnabled}},
	}, "91是质数吗？只回答是或否。")

	thought, message := joined(events, "thought"), joined(events, "message")
	if thought == "" {
		t.Error("thought channel is empty, reasoning never reached agent_thought_chunk")
	}
	if message == "" {
		t.Error("message channel is empty")
	}
	if strings.Contains(message, thought) && thought != "" {
		t.Errorf("reasoning leaked into the message channel: %q", message)
	}

	thoughtAt, messageAt := firstIndexOf(events, "thought"), firstIndexOf(events, "message")
	t.Logf("events (%d): thought@%d message@%d", len(events), thoughtAt, messageAt)
	for i, e := range events {
		s := e.Text
		if len([]rune(s)) > 24 {
			s = string([]rune(s)[:24]) + "…"
		}
		t.Logf("  #%d %-7s %q", i, e.Channel, s)
	}
	for i, e := range events {
		if e.Channel == "thought" && messageAt >= 0 && i > messageAt {
			t.Errorf("thought chunk after the answer started: event %d = %+v", i, e)
		}
	}
	checkMessageIDs(t, events)
	t.Logf("thought = %q", thought)
	t.Logf("message = %q", message)
}

// TestRunTurnCarriesHistoryAcrossPrompts sends two prompts on one session and
// inspects the requests the model received: the second one must replay the
// first exchange (question and answer) ahead of the new question, which is
// what makes the session a conversation instead of two unrelated turns.
func TestRunTurnCarriesHistoryAcrossPrompts(t *testing.T) {
	t.Setenv("HOME", t.TempDir()) // keep the session store out of the real ~/.gca
	var (
		mu        sync.Mutex
		reqBodies []string
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		reqBodies = append(reqBodies, string(body))
		mu.Unlock()

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
	defer srv.Close()

	t.Setenv("DEEPSEEK_BASE_URL", srv.URL)
	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))

	ag, clientConn, agentConn, _ := openAgentPipes(t, g, Model{
		Ref:   "deepseek/deepseek-flash",
		Name:  "deepseek-flash",
		Group: "deepseek",
	})

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	sess, err := clientConn.NewSession(ctx, acp.NewSessionRequest{Cwd: "/tmp", McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	// The first turn names its user-message id; the second leaves it to the
	// agent, and both must come back echoed in the prompt response.
	givenID := "11111111-2222-3333-4444-555555555555"
	first, err := clientConn.Prompt(ctx, acp.PromptRequest{
		SessionId: sess.SessionId,
		MessageId: &givenID,
		Prompt:    []acp.ContentBlock{acp.TextBlock("what is the first question")},
	})
	if err != nil {
		t.Fatalf("first prompt: %v", err)
	}
	if first.UserMessageId == nil || *first.UserMessageId != givenID {
		t.Fatalf("userMessageId = %v, want echoed %q", first.UserMessageId, givenID)
	}
	second, err := clientConn.Prompt(ctx, acp.PromptRequest{
		SessionId: sess.SessionId,
		Prompt:    []acp.ContentBlock{acp.TextBlock("what is the second question")},
	})
	if err != nil {
		t.Fatalf("second prompt: %v", err)
	}
	if second.UserMessageId == nil || *second.UserMessageId == "" {
		t.Fatal("second prompt response has no assigned userMessageId")
	}
	if *second.UserMessageId == givenID {
		t.Errorf("assigned userMessageId = %q, want a fresh id", *second.UserMessageId)
	}

	_ = agentConn.Done()
	_ = clientConn.Done()

	// The stored history carries both ids in genkit's metadata field: the
	// user message keeps the id it was prompted with, the answer the msg_
	// id its chunks streamed under.
	ag.mu.Lock()
	var history []*ai.Message
	if s := ag.sessions[string(sess.SessionId)]; s != nil {
		history = s.transcript
	}
	ag.mu.Unlock()
	if len(history) != 4 {
		t.Fatalf("history messages = %d, want 4 (two turns)", len(history))
	}
	wantMeta := []string{givenID, "msg_", *second.UserMessageId, "msg_"}
	for i, want := range wantMeta {
		got, _ := history[i].Metadata["messageId"].(string)
		if want == "msg_" {
			if !strings.HasPrefix(got, "msg_") {
				t.Errorf("history[%d] messageId = %q, want msg_ prefix", i, got)
			}
			continue
		}
		if got != want {
			t.Errorf("history[%d] messageId = %q, want %q", i, got, want)
		}
	}

	mu.Lock()
	defer mu.Unlock()
	if len(reqBodies) != 2 {
		t.Fatalf("model requests = %d, want 2", len(reqBodies))
	}
	req1, req2 := reqBodies[0], reqBodies[1]
	if strings.Contains(req1, "what is the second question") {
		t.Errorf("first request already carries the second question: %s", req1)
	}
	// The replayed order is what matters: prior question, prior answer, then
	// the new question, all inside one messages array.
	q1, ans, q2 := strings.Index(req2, "what is the first question"),
		strings.Index(req2, "the answer"),
		strings.Index(req2, "what is the second question")
	if q1 < 0 || ans < 0 || q2 < 0 {
		t.Fatalf("second request missing history: q1=%d ans=%d q2=%d\n%s", q1, ans, q2, req2)
	}
	if !(q1 < ans && ans < q2) {
		t.Errorf("history order = q1@%d ans@%d q2@%d, want first question, then answer, then second question", q1, ans, q2)
	}
}
