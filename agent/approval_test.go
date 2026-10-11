package agent

import (
	"context"
	"encoding/json/v2"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/deepseek"
)

// startRoleToolModel serves the fake model the remembered-choice tests
// drive: a request whose last message is from the user asks for a tool call
// on toolName carrying toolCallArg, anything later answers finalText. Each
// prompt's first request ends with the user's message, so every prompt asks
// again — which is what a second prompt needs to show that an always choice
// answered on the first never reaches the client on the next.
func startRoleToolModel(t *testing.T, toolName, toolCallArg, finalText string) func() []string {
	t.Helper()

	var (
		mu     sync.Mutex
		bodies []string
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, string(body))
		mu.Unlock()

		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"object":"list","data":[]}`)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		chunks := finalAnswerChunks(finalText)
		if lastMessageIsUser(body) {
			chunks = firstToolCallChunks(toolName, toolCallArg)
		}
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			if flusher != nil {
				flusher.Flush()
			}
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)
	t.Setenv("DEEPSEEK_BASE_URL", srv.URL)

	return func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), bodies...)
	}
}

// lastMessageIsUser reports whether the recorded chat request ends on a
// user message, the shape that asks the model for a first action.
func lastMessageIsUser(body []byte) bool {
	var req struct {
		Messages []struct {
			Role string `json:"role"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil || len(req.Messages) == 0 {
		return false
	}
	return req.Messages[len(req.Messages)-1].Role == "user"
}

// TestToolApprovalAsksBeforeRunning drives an approved tool round through
// the real ACP pipes: the gate holds the write, the client answers
// allow-once over session/request_permission, and the restart runs the tool
// under the same id — pending at the creating update, in_progress as it
// runs, completed with what it wrote, and the turn finishes with the tool's
// result in the model's next request.
func TestToolApprovalAsksBeforeRunning(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cwd := t.TempDir()
	wantPath := filepath.Join(cwd, "out.txt")

	requests := startToolModel(t, "write",
		fmt.Sprintf(`{"filePath":%q,"content":"hello"}`, wantPath), "the file is written")

	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))
	_, clientConn, agentConn, rc := openAgentPipesCaps(t, g, fakeModel(), acp.ClientCapabilities{
		Fs: acp.FileSystemCapabilities{WriteTextFile: true},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	sess, err := clientConn.NewSession(ctx, acp.NewSessionRequest{Cwd: cwd, McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	if _, err := clientConn.Prompt(ctx, acp.PromptRequest{
		SessionId: sess.SessionId,
		Prompt:    []acp.ContentBlock{acp.TextBlock("write hello to out.txt")},
	}); err != nil {
		t.Fatalf("prompt: %v", err)
	}
	_ = agentConn.Done()
	_ = clientConn.Done()

	// One prompt went out, on this session, offering all four outcomes.
	perms := rc.permissionRequests()
	if len(perms) != 1 {
		t.Fatalf("permission requests = %d (%+v), want exactly one", len(perms), perms)
	}
	if perms[0].SessionId != sess.SessionId {
		t.Errorf("sessionId = %q, want the session the turn ran on %q", perms[0].SessionId, sess.SessionId)
	}
	kinds := map[acp.PermissionOptionKind]int{}
	for _, opt := range perms[0].Options {
		kinds[opt.Kind]++
	}
	for _, want := range []acp.PermissionOptionKind{
		acp.PermissionOptionKindAllowOnce,
		acp.PermissionOptionKindAllowAlways,
		acp.PermissionOptionKindRejectOnce,
		acp.PermissionOptionKindRejectAlways,
	} {
		if kinds[want] != 1 {
			t.Errorf("option kind %q offered %d times, want once (options: %+v)", want, kinds[want], perms[0].Options)
		}
	}

	// The prompt names the call the creating update filed: same toolCallId,
	// from pending through to completed.
	events := waitForToolEvents(t, rc, 3)
	create := events[0].ToolCall
	if create == nil || create.Status != acp.ToolCallStatusPending {
		t.Fatalf("first update = %+v, want the pending creating update", events[0])
	}
	if string(perms[0].ToolCall.ToolCallId) != string(create.ToolCallId) {
		t.Errorf("permission toolCallId = %q, want the creating id %q", perms[0].ToolCall.ToolCallId, create.ToolCallId)
	}
	if s := statusOf(events[1]); s != acp.ToolCallStatusInProgress {
		t.Errorf("second status = %q, want in_progress", s)
	}
	if s := statusOf(events[2]); s != acp.ToolCallStatusCompleted {
		t.Errorf("third status = %q, want completed", s)
	}

	// Approved means run: the write went over fs/write_text_file.
	writes := rc.fsWrites()
	if len(writes) != 1 || writes[0].Path != wantPath || writes[0].Content != "hello" {
		t.Fatalf("fs/write_text_file requests = %+v, want one for %q", writes, wantPath)
	}

	// And the model saw what it did: two requests, the second carrying the
	// tool's result, so the turn could finish.
	bodies := requests()
	if len(bodies) != 2 {
		t.Fatalf("model requests = %d, want 2 (tool call round)", len(bodies))
	}
	if !strings.Contains(bodies[1], "Wrote 5 bytes") {
		t.Errorf("second request missing the tool result:\n%s", bodies[1])
	}
}

// TestToolApprovalRejectOnceSkipsTool drives the refusal: reject-once never
// runs the tool, closes the client's call as failed with the reason, and
// answers the model with a response it can read — the turn still finishes,
// on a conversation nothing is left hanging in.
func TestToolApprovalRejectOnceSkipsTool(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cwd := t.TempDir()
	wantPath := filepath.Join(cwd, "out.txt")

	requests := startToolModel(t, "write",
		fmt.Sprintf(`{"filePath":%q,"content":"hello"}`, wantPath), "the file is refused")

	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))
	_, clientConn, agentConn, rc := openAgentPipesCaps(t, g, fakeModel(), acp.ClientCapabilities{
		Fs: acp.FileSystemCapabilities{WriteTextFile: true},
	})
	rc.permission = func(context.Context, acp.RequestPermissionRequest) (acp.RequestPermissionResponse, error) {
		return selectPermission(permissionRejectOnce), nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	sess, err := clientConn.NewSession(ctx, acp.NewSessionRequest{Cwd: cwd, McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	if _, err := clientConn.Prompt(ctx, acp.PromptRequest{
		SessionId: sess.SessionId,
		Prompt:    []acp.ContentBlock{acp.TextBlock("write hello to out.txt")},
	}); err != nil {
		t.Fatalf("prompt: %v", err)
	}
	_ = agentConn.Done()
	_ = clientConn.Done()

	if perms := rc.permissionRequests(); len(perms) != 1 {
		t.Fatalf("permission requests = %d, want exactly one", len(perms))
	}

	// The call closed as failed under the creating id, never in_progress:
	// the gate sat outside the reporter, so nothing ran to report.
	events := waitForToolEvents(t, rc, 2)
	create := events[0].ToolCall
	if create == nil || create.Status != acp.ToolCallStatusPending {
		t.Fatalf("first update = %+v, want the pending creating update", events[0])
	}
	failed := events[1].ToolCallUpdate
	if failed == nil || failed.ToolCallId != create.ToolCallId {
		t.Fatalf("second update = %+v, want one on %q", events[1], create.ToolCallId)
	}
	if failed.Status == nil || *failed.Status != acp.ToolCallStatusFailed {
		t.Errorf("status = %v, want failed", failed.Status)
	}
	if failed.RawOutput != denialReply {
		t.Errorf("rawOutput = %#v, want the denial the model read", failed.RawOutput)
	}

	if writes := rc.fsWrites(); len(writes) != 0 {
		t.Errorf("fs/write_text_file requests = %+v, want none from a refused call", writes)
	}

	bodies := requests()
	if len(bodies) != 2 {
		t.Fatalf("model requests = %d, want 2 (the refusal still answers the call)", len(bodies))
	}
	if !strings.Contains(bodies[1], denialReply) {
		t.Errorf("second request missing the denial:\n%s", bodies[1])
	}
	if !strings.Contains(bodies[1], wantPath) {
		t.Errorf("second request's refusal missing the path:\n%s", bodies[1])
	}
}

// TestToolApprovalAllowAlwaysSticksToSession drives two prompts on one
// session with an always-allow on the first: the second prompt's write runs
// without the client ever being asked again, for the gate carries the
// session's remembered choice into every call of every turn. Nothing is
// written anywhere: the memory dies with the session.
func TestToolApprovalAllowAlwaysSticksToSession(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cwd := t.TempDir()

	requests := startRoleToolModel(t, "write",
		fmt.Sprintf(`{"filePath":%q,"content":"hello"}`, filepath.Join(cwd, "out.txt")), "done")

	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))
	_, clientConn, agentConn, rc := openAgentPipesCaps(t, g, fakeModel(), acp.ClientCapabilities{
		Fs: acp.FileSystemCapabilities{WriteTextFile: true},
	})
	rc.permission = func(context.Context, acp.RequestPermissionRequest) (acp.RequestPermissionResponse, error) {
		return selectPermission(permissionAllowAlways), nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	sess, err := clientConn.NewSession(ctx, acp.NewSessionRequest{Cwd: cwd, McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	prompt := func(text string) {
		t.Helper()
		if _, err := clientConn.Prompt(ctx, acp.PromptRequest{
			SessionId: sess.SessionId,
			Prompt:    []acp.ContentBlock{acp.TextBlock(text)},
		}); err != nil {
			t.Fatalf("prompt %q: %v", text, err)
		}
	}

	prompt("first write")
	if perms := rc.permissionRequests(); len(perms) != 1 {
		t.Fatalf("permission requests after first prompt = %d, want one", len(perms))
	}
	if writes := rc.fsWrites(); len(writes) != 1 {
		t.Fatalf("fs writes after first prompt = %d, want one", len(writes))
	}
	if bodies := requests(); len(bodies) != 2 {
		t.Fatalf("model requests after first prompt = %d, want the tool round closed", len(bodies))
	}

	prompt("second write")
	if perms := rc.permissionRequests(); len(perms) != 1 {
		t.Fatalf("permission requests after second prompt = %d, want the always-allow to spare it", len(perms))
	}
	if writes := rc.fsWrites(); len(writes) != 2 {
		t.Fatalf("fs writes after second prompt = %d, want the second write to have run", len(writes))
	}
	if bodies := requests(); len(bodies) != 4 {
		t.Fatalf("model requests after second prompt = %d, want 4", len(bodies))
	}

	_ = agentConn.Done()
	_ = clientConn.Done()
}

// TestToolApprovalRejectAlwaysSkipsPrompt drives the mirror: reject-always
// answers the first prompt, and the next request for the same tool is
// refused from the session's memory — one prompt in the whole session, no
// write in either round, and the model told no both times.
func TestToolApprovalRejectAlwaysSkipsPrompt(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cwd := t.TempDir()

	requests := startRoleToolModel(t, "write",
		fmt.Sprintf(`{"filePath":%q,"content":"hello"}`, filepath.Join(cwd, "out.txt")), "done")

	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))
	_, clientConn, agentConn, rc := openAgentPipesCaps(t, g, fakeModel(), acp.ClientCapabilities{
		Fs: acp.FileSystemCapabilities{WriteTextFile: true},
	})
	rc.permission = func(context.Context, acp.RequestPermissionRequest) (acp.RequestPermissionResponse, error) {
		return selectPermission(permissionRejectAlways), nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	sess, err := clientConn.NewSession(ctx, acp.NewSessionRequest{Cwd: cwd, McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	prompt := func(text string) {
		t.Helper()
		if _, err := clientConn.Prompt(ctx, acp.PromptRequest{
			SessionId: sess.SessionId,
			Prompt:    []acp.ContentBlock{acp.TextBlock(text)},
		}); err != nil {
			t.Fatalf("prompt %q: %v", text, err)
		}
	}

	prompt("first write")
	if perms := rc.permissionRequests(); len(perms) != 1 {
		t.Fatalf("permission requests after first prompt = %d, want one", len(perms))
	}
	bodies := requests()
	if len(bodies) != 2 || !strings.Contains(bodies[1], denialReply) {
		t.Fatalf("model requests after first prompt = %d, want the round closed on the denial", len(bodies))
	}

	prompt("second write")
	if perms := rc.permissionRequests(); len(perms) != 1 {
		t.Fatalf("permission requests after second prompt = %d, want the always-reject to spare the prompt", len(perms))
	}
	bodies = requests()
	if len(bodies) != 4 {
		t.Fatalf("model requests after second prompt = %d, want 4", len(bodies))
	}
	if !strings.Contains(bodies[3], denialReply) {
		t.Errorf("fourth request missing the denial:\n%s", bodies[3])
	}
	if writes := rc.fsWrites(); len(writes) != 0 {
		t.Errorf("fs/write_text_file requests = %+v, want none ever", writes)
	}

	_ = agentConn.Done()
	_ = clientConn.Done()
}

// TestToolApprovalResumesSeveralInterruptsAtOnce drives one model response
// asking for two tools: both are held, both get their own permission prompt
// under the ids their creating updates were filed with, and one resume
// restarts them together — two calls, two rounds of lifecycle updates, one
// finished conversation.
func TestToolApprovalResumesSeveralInterruptsAtOnce(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cwd := t.TempDir()
	writePath := filepath.Join(cwd, "out.txt")

	var (
		mu     sync.Mutex
		bodies []string
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, string(body))
		n := len(bodies)
		mu.Unlock()

		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"object":"list","data":[]}`)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		chunks := finalAnswerChunks("both done")
		if n == 1 {
			writeArgs := fmt.Sprintf(`{"filePath":%q,"content":"hello"}`, writePath)
			readArgs := fmt.Sprintf(`{"filePath":%q}`, filepath.Join("src", "main.go"))
			chunks = []string{
				fmt.Sprintf(`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"write","arguments":%q}},{"index":1,"id":"call_2","type":"function","function":{"name":"read","arguments":%q}}]},"finish_reason":null}]}`, writeArgs, readArgs),
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
			}
		}
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			if flusher != nil {
				flusher.Flush()
			}
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)
	t.Setenv("DEEPSEEK_BASE_URL", srv.URL)

	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))
	_, clientConn, agentConn, rc := openAgentPipesCaps(t, g, fakeModel(), acp.ClientCapabilities{
		Fs: acp.FileSystemCapabilities{ReadTextFile: true, WriteTextFile: true},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	sess, err := clientConn.NewSession(ctx, acp.NewSessionRequest{Cwd: cwd, McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	if _, err := clientConn.Prompt(ctx, acp.PromptRequest{
		SessionId: sess.SessionId,
		Prompt:    []acp.ContentBlock{acp.TextBlock("write then read")},
	}); err != nil {
		t.Fatalf("prompt: %v", err)
	}
	_ = agentConn.Done()
	_ = clientConn.Done()

	// One prompt per interrupt, each naming one of the creating updates.
	perms := rc.permissionRequests()
	if len(perms) != 2 {
		t.Fatalf("permission requests = %d (%+v), want one per held tool", len(perms), perms)
	}
	events := waitForToolEvents(t, rc, 6)
	created := map[acp.ToolCallId]bool{}
	counts := map[acp.ToolCallStatus]int{}
	for _, ev := range events {
		if ev.ToolCall != nil {
			created[ev.ToolCall.ToolCallId] = true
			counts[ev.ToolCall.Status]++
			continue
		}
		if ev.ToolCallUpdate != nil && ev.ToolCallUpdate.Status != nil {
			counts[*ev.ToolCallUpdate.Status]++
		}
	}
	if counts[acp.ToolCallStatusPending] != 2 || counts[acp.ToolCallStatusInProgress] != 2 || counts[acp.ToolCallStatusCompleted] != 2 {
		t.Errorf("status counts = %v, want two of pending, in_progress, completed", counts)
	}
	for i, p := range perms {
		if !created[p.ToolCall.ToolCallId] {
			t.Errorf("permission %d names toolCallId %q, want one of the creating updates", i, p.ToolCall.ToolCallId)
		}
	}
	if perms[0].ToolCall.ToolCallId == perms[1].ToolCall.ToolCallId {
		t.Errorf("both permissions name %q, want one per tool", perms[0].ToolCall.ToolCallId)
	}

	// One resume ran both: the write reached the client, the read's answer
	// came back from it, and the model's second request carries both.
	if writes := rc.fsWrites(); len(writes) != 1 || writes[0].Path != writePath {
		t.Errorf("fs/write_text_file requests = %+v, want one for %q", writes, writePath)
	}
	if reads := rc.fsReads(); len(reads) != 1 {
		t.Errorf("fs/read_text_file requests = %+v, want one", reads)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(bodies) != 2 {
		t.Fatalf("model requests = %d, want 2", len(bodies))
	}
	if !strings.Contains(bodies[1], "Wrote 5 bytes") || !strings.Contains(bodies[1], "file body from client") {
		t.Errorf("second request missing one of the two results:\n%s", bodies[1])
	}
}

// TestToolApprovalCancelledPermissionEndsTurn covers the protocol's third
// outcome: a cancelled permission response means the prompt turn itself was
// cancelled, so the turn stops without an answer, without running the tool,
// and without storing anything for a later prompt to replay.
func TestToolApprovalCancelledPermissionEndsTurn(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cwd := t.TempDir()

	requests := startToolModel(t, "write",
		fmt.Sprintf(`{"filePath":%q,"content":"hello"}`, filepath.Join(cwd, "out.txt")), "never reached")

	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))
	ag, clientConn, agentConn, rc := openAgentPipesCaps(t, g, fakeModel(), acp.ClientCapabilities{
		Fs: acp.FileSystemCapabilities{WriteTextFile: true},
	})
	rc.permission = func(context.Context, acp.RequestPermissionRequest) (acp.RequestPermissionResponse, error) {
		return acp.RequestPermissionResponse{
			Outcome: acp.RequestPermissionOutcome{
				Cancelled: &acp.RequestPermissionOutcomeCancelled{Outcome: "cancelled"},
			},
		}, nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	sess, err := clientConn.NewSession(ctx, acp.NewSessionRequest{Cwd: cwd, McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	resp, err := clientConn.Prompt(ctx, acp.PromptRequest{
		SessionId: sess.SessionId,
		Prompt:    []acp.ContentBlock{acp.TextBlock("write hello to out.txt")},
	})
	if err != nil {
		t.Fatalf("prompt: %v", err)
	}
	if resp.StopReason != acp.StopReasonCancelled {
		t.Errorf("stopReason = %q, want cancelled", resp.StopReason)
	}
	_ = agentConn.Done()
	_ = clientConn.Done()

	if writes := rc.fsWrites(); len(writes) != 0 {
		t.Errorf("fs/write_text_file requests = %+v, want none from a cancelled turn", writes)
	}
	if bodies := requests(); len(bodies) != 1 {
		t.Errorf("model requests = %d, want 1 — the resume never ran", len(bodies))
	}

	// Nothing was stored: the next prompt on this session would start
	// over rather than replay a half-answered turn.
	s, ok := ag.get(string(sess.SessionId))
	if !ok {
		t.Fatal("session not registered")
	}
	stored, _, err := s.load(string(sess.SessionId))
	if err != nil {
		t.Fatalf("read stored transcript: %v", err)
	}
	if len(stored) != 0 {
		t.Errorf("stored transcript = %d messages, want none", len(stored))
	}
}
