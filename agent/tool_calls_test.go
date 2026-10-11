package agent

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/deepseek"
)

// TestToolCallStartFilesCallUnderItsRef checks the creating update built for
// one tool request without a model in the way: the id is the request's own
// Ref, so a replay rebuilds the call under the id the live turn reported it
// with, the status is the pending one nothing has run under yet, the title
// and kind read from the tool, and the file the call names shows up as a
// location resolved to an absolute path. The raw input must be the model's
// own map, unmodified — the update is metadata about the call, not a rewrite
// of what the tool will receive.
func TestToolCallStartFilesCallUnderItsRef(t *testing.T) {
	cwd := "/workspace"
	line := 7
	cases := []struct {
		name      string
		req       *ai.ToolRequest
		wantTitle string
		wantKind  acp.ToolKind
		wantPath  string
		wantLine  *int
	}{
		{
			name: "read relative path with line",
			req: &ai.ToolRequest{Ref: "call_1", Name: "read", Input: map[string]any{
				"filePath": "src/main.go",
				"line":     float64(line),
			}},
			wantTitle: "Reading /workspace/src/main.go",
			wantKind:  acp.ToolKindRead,
			wantPath:  "/workspace/src/main.go",
			wantLine:  &line,
		},
		{
			name: "write absolute path",
			req: &ai.ToolRequest{Ref: "call_2", Name: "write", Input: map[string]any{
				"filePath": "/abs/out.txt",
				"content":  "x",
			}},
			wantTitle: "Writing /abs/out.txt",
			wantKind:  acp.ToolKindEdit,
			wantPath:  "/abs/out.txt",
		},
		{
			name:      "read without a path",
			req:       &ai.ToolRequest{Ref: "call_3", Name: "read", Input: map[string]any{}},
			wantTitle: "Reading file",
			wantKind:  acp.ToolKindRead,
		},
		{
			name:      "unknown tool falls back to its own name",
			req:       &ai.ToolRequest{Ref: "call_4", Name: "search", Input: map[string]any{"query": "x"}},
			wantTitle: "search",
			wantKind:  acp.ToolKindOther,
		},
		{
			name: "use_skill titles the skill it loads, not a path",
			req: &ai.ToolRequest{Ref: "call_5", Name: useSkillToolName, Input: map[string]any{
				"skillName": "python-helper",
			}},
			wantTitle: "Loading skill python-helper",
			wantKind:  acp.ToolKindOther,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			original, _ := tc.req.Input.(map[string]any)
			asStored := tc.req.Ref
			update := toolCallStart(tc.req, cwd)

			toolCall := update.ToolCall
			if toolCall == nil {
				t.Fatal("update carries no tool_call")
			}
			if toolCall.Status != acp.ToolCallStatusPending {
				t.Errorf("status = %q, want pending", toolCall.Status)
			}
			if toolCall.Title != tc.wantTitle {
				t.Errorf("title = %q, want %q", toolCall.Title, tc.wantTitle)
			}
			if toolCall.Kind != tc.wantKind {
				t.Errorf("kind = %q, want %q", toolCall.Kind, tc.wantKind)
			}

			// The stored ref is the id: a replay files the call under the
			// one the live turn gave the client, and the builder must not
			// mint a second one of its own.
			if id := string(toolCall.ToolCallId); id != asStored {
				t.Errorf("toolCallId = %q, want the request's own %q", id, asStored)
			}
			if tc.req.Ref != asStored {
				t.Errorf("request Ref = %q, want it left as stored", tc.req.Ref)
			}

			if tc.wantPath == "" {
				if len(toolCall.Locations) != 0 {
					t.Errorf("locations = %+v, want none", toolCall.Locations)
				}
			} else {
				if len(toolCall.Locations) != 1 || toolCall.Locations[0].Path != tc.wantPath {
					t.Fatalf("locations = %+v, want one entry for %q", toolCall.Locations, tc.wantPath)
				}
				gotLine := toolCall.Locations[0].Line
				if tc.wantLine == nil {
					if gotLine != nil {
						t.Errorf("line = %d, want none", *gotLine)
					}
				} else if gotLine == nil || *gotLine != *tc.wantLine {
					t.Errorf("line = %v, want %d", gotLine, *tc.wantLine)
				}
			}

			// rawInput is the model's arguments as sent: no injected path
			// key, no dropped field, same map the tool will decode.
			raw, ok := toolCall.RawInput.(map[string]any)
			if !ok {
				t.Fatalf("rawInput = %#v, want the decoded argument map", toolCall.RawInput)
			}
			if _, injected := raw["path"]; injected {
				t.Errorf("rawInput gained a path key: %#v", raw)
			}
			for k, v := range original {
				if raw[k] != v {
					t.Errorf("rawInput[%q] = %#v, want %#v", k, raw[k], v)
				}
			}
		})
	}
}

// TestTurnReportsToolCallCreation drives a tool round through the real ACP
// pipes and checks the creating end as the client sees it: exactly one
// tool_call update (the streamed tool-call delta must not report the call a
// second time), pending, naming the file the model asked for, and sent
// before the read went over the wire.
func TestTurnReportsToolCallCreation(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cwd := t.TempDir()
	wantPath := filepath.Join(cwd, "src", "main.go")
	relative := filepath.Join("src", "main.go")

	startToolModel(t, "read", fmt.Sprintf(`{"filePath":%q}`, relative), "the file is read")

	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))
	_, clientConn, agentConn, rc := openAgentPipesCaps(t, g, fakeModel(), acp.ClientCapabilities{
		Fs: acp.FileSystemCapabilities{ReadTextFile: true},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	sess, err := clientConn.NewSession(ctx, acp.NewSessionRequest{Cwd: cwd, McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	if _, err := clientConn.Prompt(ctx, acp.PromptRequest{
		SessionId: sess.SessionId,
		Prompt:    []acp.ContentBlock{acp.TextBlock("show me main.go")},
	}); err != nil {
		t.Fatalf("prompt: %v", err)
	}
	_ = agentConn.Done()
	_ = clientConn.Done()

	updates := rc.toolCallUpdates()
	if len(updates) != 1 {
		t.Fatalf("tool_call updates = %d (%+v), want exactly one", len(updates), updates)
	}
	toolCall := updates[0]
	if !strings.HasPrefix(string(toolCall.ToolCallId), "call_") {
		t.Errorf("toolCallId = %q, want call_ prefix", toolCall.ToolCallId)
	}
	// The provider's own id is renumbered out: the id the client is given
	// is the agent's, and that is the one the transcript keeps for a later
	// replay to rebuild the call under.
	if string(toolCall.ToolCallId) == "call_1" {
		t.Errorf("toolCallId = %q, want a fresh id rather than the provider's", toolCall.ToolCallId)
	}
	if toolCall.Status != acp.ToolCallStatusPending {
		t.Errorf("status = %q, want pending", toolCall.Status)
	}
	if toolCall.Kind != acp.ToolKindRead {
		t.Errorf("kind = %q, want read", toolCall.Kind)
	}
	if want := "Reading " + wantPath; toolCall.Title != want {
		t.Errorf("title = %q, want %q", toolCall.Title, want)
	}
	if len(toolCall.Locations) != 1 || toolCall.Locations[0].Path != wantPath {
		t.Errorf("locations = %+v, want one entry for %q", toolCall.Locations, wantPath)
	}
	raw, ok := toolCall.RawInput.(map[string]any)
	if !ok || raw["filePath"] != relative {
		t.Errorf("rawInput = %#v, want the model's own arguments", toolCall.RawInput)
	}

	// The create is not a stand-in for the call: the read still went over
	// fs/read_text_file, after the client had been told about it.
	reads := rc.fsReads()
	if len(reads) != 1 || reads[0].Path != wantPath {
		t.Errorf("fs/read_text_file requests = %+v, want one for %q", reads, wantPath)
	}
}

// TestTurnReportsToolCallLifecycle drives a tool round and checks the whole
// sequence the client sees the call in: the creating update pending, then
// the same id going in_progress as the tool runs, then completed carrying
// the output the model read back. Three reports, one toolCallId.
func TestTurnReportsToolCallLifecycle(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cwd := t.TempDir()
	relative := filepath.Join("src", "main.go")

	startToolModel(t, "read", fmt.Sprintf(`{"filePath":%q}`, relative), "the file is read")

	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))
	_, clientConn, agentConn, rc := openAgentPipesCaps(t, g, fakeModel(), acp.ClientCapabilities{
		Fs: acp.FileSystemCapabilities{ReadTextFile: true},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	sess, err := clientConn.NewSession(ctx, acp.NewSessionRequest{Cwd: cwd, McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	if _, err := clientConn.Prompt(ctx, acp.PromptRequest{
		SessionId: sess.SessionId,
		Prompt:    []acp.ContentBlock{acp.TextBlock("show me main.go")},
	}); err != nil {
		t.Fatalf("prompt: %v", err)
	}
	_ = agentConn.Done()
	_ = clientConn.Done()

	events := rc.toolCallEvents()
	if len(events) != 3 {
		t.Fatalf("tool call updates = %d (%+v), want creating, in_progress, completed", len(events), events)
	}

	create := events[0].ToolCall
	if create == nil || create.Status != acp.ToolCallStatusPending {
		t.Fatalf("first update = %+v, want the pending creating update", events[0])
	}
	id := create.ToolCallId

	start := events[1].ToolCallUpdate
	if start == nil {
		t.Fatalf("second update = %+v, want a tool_call_update", events[1])
	}
	if start.ToolCallId != id {
		t.Errorf("second toolCallId = %q, want the creating id %q", start.ToolCallId, id)
	}
	if start.Status == nil || *start.Status != acp.ToolCallStatusInProgress {
		t.Errorf("status = %v, want in_progress", start.Status)
	}

	done := events[2].ToolCallUpdate
	if done == nil {
		t.Fatalf("third update = %+v, want a tool_call_update", events[2])
	}
	if done.ToolCallId != id {
		t.Errorf("third toolCallId = %q, want the creating id %q", done.ToolCallId, id)
	}
	if done.Status == nil || *done.Status != acp.ToolCallStatusCompleted {
		t.Errorf("status = %v, want completed", done.Status)
	}
	// The result is the output the model read back, not a re-run of the
	// tool: whatever the client's fs/read_text_file answered with.
	if done.RawOutput != "file body from client" {
		t.Errorf("rawOutput = %#v, want the output the model saw", done.RawOutput)
	}
}

// TestWrapToolReportsFailure drives the hook directly with a tool that
// errors: the client hears in_progress first and failed after, under the
// request's own id with the error as the raw output, and the caller still
// gets the tool's error rather than one from the reporting. A request
// carrying no id reports nothing and runs anyway: there is no call on the
// client to close.
func TestWrapToolReportsFailure(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	ag, _, _, rc := openAgentPipes(t, nil, fakeModel())

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	hooks, err := toolCallReporter{agent: ag, sid: "sess_x", cwd: "/workspace"}.New(ctx)
	if err != nil {
		t.Fatalf("new hooks: %v", err)
	}
	if hooks.WrapTool == nil {
		t.Fatal("hooks carry no WrapTool")
	}

	boom := errors.New("boom")
	params := &ai.ToolParams{Request: &ai.ToolRequest{
		Ref:   "call_fail",
		Name:  "read",
		Input: map[string]any{"filePath": "main.go"},
	}}
	_, err = hooks.WrapTool(ctx, params, func(context.Context, *ai.ToolParams) (*ai.MultipartToolResponse, error) {
		return nil, boom
	})
	if !errors.Is(err, boom) {
		t.Fatalf("wrapTool error = %v, want the tool's own %v", err, boom)
	}

	// The notifications cross the pipe asynchronously: wait for both to
	// reach the client rather than reading what happens to have arrived.
	events := waitForToolEvents(t, rc, 2)
	if len(events) != 2 {
		t.Fatalf("tool call updates = %d (%+v), want in_progress and failed", len(events), events)
	}
	if s := statusOf(events[0]); s != acp.ToolCallStatusInProgress {
		t.Errorf("first status = %q, want in_progress", s)
	}
	failed := events[1].ToolCallUpdate
	if failed == nil || failed.ToolCallId != "call_fail" {
		t.Fatalf("second update = %+v, want one on call_fail", events[1])
	}
	if failed.Status == nil || *failed.Status != acp.ToolCallStatusFailed {
		t.Errorf("status = %v, want failed", failed.Status)
	}
	if failed.RawOutput != "boom" {
		t.Errorf("rawOutput = %#v, want the error text", failed.RawOutput)
	}

	// No id, no report — and the tool still runs.
	idless, err := hooks.WrapTool(ctx, &ai.ToolParams{Request: &ai.ToolRequest{Name: "read"}},
		func(context.Context, *ai.ToolParams) (*ai.MultipartToolResponse, error) {
			return &ai.MultipartToolResponse{Output: "ran"}, nil
		})
	if err != nil || idless == nil || idless.Output != "ran" {
		t.Fatalf("idless run = (%v, %v), want the tool's own response", idless, err)
	}
	if got := len(rc.toolCallEvents()); got != 2 {
		t.Errorf("tool call updates after the idless run = %d, want still 2", got)
	}
}

// waitForToolEvents blocks until the client has seen at least n tool
// updates, and fails the test when they do not arrive: session/update
// notifications cross the pipe asynchronously, so a hook that just returned
// may not have been dispatched yet.
func waitForToolEvents(t *testing.T, rc *recordingClient, n int) []acp.SessionUpdate {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for {
		events := rc.toolCallEvents()
		if len(events) >= n {
			return events
		}
		if time.Now().After(deadline) {
			t.Fatalf("tool call updates = %d, want at least %d", len(events), n)
		}
		time.Sleep(time.Millisecond)
	}
}

// statusOf reads the status off whichever tool update shape ev carries.
func statusOf(ev acp.SessionUpdate) acp.ToolCallStatus {
	if ev.ToolCall != nil {
		return ev.ToolCall.Status
	}
	if ev.ToolCallUpdate != nil && ev.ToolCallUpdate.Status != nil {
		return *ev.ToolCallUpdate.Status
	}
	return ""
}
