package agent

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/deepseek"
)

// firstToolCallChunks streams a single tool call asking for the named tool.
func firstToolCallChunks(toolName, args string) []string {
	return []string{
		fmt.Sprintf(`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":%q,"arguments":%q}}]},"finish_reason":null}]}`, toolName, args),
		`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
	}
}

func finalAnswerChunks(text string) []string {
	return []string{
		fmt.Sprintf(`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{"role":"assistant","content":%q},"finish_reason":null}]}`, text),
		`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25}}`,
	}
}

// startToolModelCalls serves the fake model the tool tests drive: call is
// asked for the request at hand (1-based, counting chat completions) and
// answers with a tool call on the name and arguments it returns, or with
// finalText when it returns no name. It returns the request bodies it
// recorded, in arrival order, and points DEEPSEEK_BASE_URL at itself for the
// duration of the test.
func startToolModelCalls(t *testing.T, call func(n int) (tool, args string), finalText string) func() []string {
	t.Helper()

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
		chunks := finalAnswerChunks(finalText)
		if tool, args := call(n); tool != "" {
			chunks = firstToolCallChunks(tool, args)
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

// startToolModel serves the one-call variant: the first request answers with
// a tool call on toolName carrying toolCallArg, every later one with
// finalText.
func startToolModel(t *testing.T, toolName, toolCallArg, finalText string) func() []string {
	t.Helper()
	return startToolModelCalls(t, func(n int) (string, string) {
		if n == 1 {
			return toolName, toolCallArg
		}
		return "", ""
	}, finalText)
}

// seedFile plants one file under the session cwd for a read to find.
func seedFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// TestWriteFileToolWritesOnDisk drives a whole tool round through the real
// ACP pipes: the model asks write_file, the approval gate holds it, the
// default allow-once answer runs it, and the file lands on the agent's own
// disk under the session cwd — the client holds no file surface any more, so
// nothing goes over fs/write_text_file. The stored transcript is what the
// round must leave behind either way.
func TestWriteFileToolWritesOnDisk(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cwd := t.TempDir()
	wantPath := filepath.Join(cwd, "sub", "out.txt")

	toolCallArg := fmt.Sprintf(`{"filePath":%q,"content":"hello from tool"}`, filepath.Join("sub", "out.txt"))
	requests := startToolModel(t, "write_file", toolCallArg, "the file is written")

	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))

	ag, clientConn, agentConn, rc := openAgentPipes(t, g, fakeModel())

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

	// The write went to the agent's own disk, through the middleware's
	// os.Root, with the model's relative path landing under the cwd.
	data, err := os.ReadFile(wantPath)
	if err != nil {
		t.Fatalf("read written file: %v", err)
	}
	if string(data) != "hello from tool" {
		t.Errorf("file content = %q, want what the model asked for", data)
	}
	// The client's fs surface was not used, capability advertised or not:
	// ACP v2 removes those methods, and this agent no longer calls them.
	if writes := rc.fsWrites(); len(writes) != 0 {
		t.Errorf("fs/write_text_file requests = %+v, want none", writes)
	}
	if reads := rc.fsReads(); len(reads) != 0 {
		t.Errorf("fs/read_text_file requests = %+v, want none", reads)
	}

	reqBodies := requests()
	if len(reqBodies) != 2 {
		t.Fatalf("model requests = %d, want 2 (tool call round)", len(reqBodies))
	}
	// The second request must carry the tool response so the model can see
	// what its call did.
	if !strings.Contains(reqBodies[1], "File "+filepath.Join("sub", "out.txt")+" written successfully.") {
		t.Errorf("second request missing the tool result:\n%s", reqBodies[1])
	}

	// The transcript keeps the whole round: prompt, tool-request message,
	// tool response, and the final answer, all replayable next turn.
	s, ok := ag.get(string(sess.SessionId))
	if !ok {
		t.Fatal("session not registered")
	}
	stored, _, err := s.load(string(sess.SessionId))
	if err != nil {
		t.Fatalf("read stored transcript: %v", err)
	}
	if len(stored) < 4 {
		t.Fatalf("stored transcript = %d messages, want at least 4 (prompt, tool call, tool result, answer)", len(stored))
	}
	if stored[0].Text() != "write hello to out.txt" {
		t.Errorf("first stored message = %q, want the prompt", stored[0].Text())
	}
	last := stored[len(stored)-1]
	if last.Text() != "the file is written" {
		t.Errorf("last stored message = %q, want the final answer", last.Text())
	}
	gotID, _ := last.Metadata["messageId"].(string)
	if !strings.HasPrefix(gotID, "msg_") {
		t.Errorf("final messageId = %q, want msg_ prefix", gotID)
	}
	// The prompt keeps the id its turn named.
	savedID, _ := stored[0].Metadata["messageId"].(string)
	if savedID == "" || strings.HasPrefix(savedID, "msg_") {
		t.Errorf("prompt messageId = %q, want a plain UUID", savedID)
	}

	// The model never saw a system message in the stored transcript: it is
	// re-sent every turn, so storing it would double it.
	for i, m := range stored {
		if m.Role == "system" {
			t.Errorf("stored message %d has role system, want none stored", i)
		}
	}
}

// TestReadFileToolReadsFromDisk drives a read round: the model's relative
// path resolves under the session cwd, the contents it sees are the ones on
// the agent's disk — read_file answers with a summary and injects the file
// itself as a user message on the next turn — and the client is asked for
// nothing over fs/read_text_file. The injected contents must not reach the
// client as the agent's own reply either: they are the model's input, not
// the answer being streamed.
func TestReadFileToolReadsFromDisk(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cwd := t.TempDir()
	seedFile(t, filepath.Join(cwd, "src", "main.go"), "package main\n\nfunc main() {}\n")

	toolCallArg := fmt.Sprintf(`{"filePath":%q}`, filepath.Join("src", "main.go"))
	requests := startToolModel(t, "read_file", toolCallArg, "the file is read")

	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))

	_, clientConn, agentConn, rc := openAgentPipes(t, g, fakeModel())

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

	if reads := rc.fsReads(); len(reads) != 0 {
		t.Errorf("fs/read_text_file requests = %+v, want none", reads)
	}
	reqBodies := requests()
	if len(reqBodies) != 2 {
		t.Fatalf("model requests = %d, want 2 (tool call round)", len(reqBodies))
	}
	// The second request carries the summary as the tool's result and the
	// file itself as the injected user message behind it.
	if !strings.Contains(reqBodies[1], "package main") {
		t.Errorf("second request missing the file contents:\n%s", reqBodies[1])
	}
	// The body is JSON, where "<" arrives escaped as <, so the envelope is
	// matched without its opening bracket.
	if !strings.Contains(reqBodies[1], "read_file path=") {
		t.Errorf("second request missing the injected read_file envelope:\n%s", reqBodies[1])
	}

	// The contents went to the model, not to the client: the answer channel
	// carries the final reply and nothing else.
	rc.testClient.mu.Lock()
	events := append([]streamEvent(nil), rc.events...)
	rc.testClient.mu.Unlock()
	if got := joined(events, "message"); got != "the file is read" {
		t.Errorf("message channel = %q, want only the final answer", got)
	}
}

// TestFileToolsIgnoreClientCapabilities pins the migration: a client that
// still advertises the fs capabilities ACP v2 removes gets its read answered
// from the agent's own disk regardless, and none of the client's fs methods
// are ever called. What the client would have returned is what must NOT
// reach the model.
func TestFileToolsIgnoreClientCapabilities(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cwd := t.TempDir()
	seedFile(t, filepath.Join(cwd, "notes.txt"), "content from the agent's disk\n")

	requests := startToolModel(t, "read_file", `{"filePath":"notes.txt"}`, "the file is read")

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
		Prompt:    []acp.ContentBlock{acp.TextBlock("show me notes.txt")},
	}); err != nil {
		t.Fatalf("prompt: %v", err)
	}
	_ = agentConn.Done()
	_ = clientConn.Done()

	if reads := rc.fsReads(); len(reads) != 0 {
		t.Errorf("fs/read_text_file requests = %+v, want none despite the capability", reads)
	}
	if writes := rc.fsWrites(); len(writes) != 0 {
		t.Errorf("fs/write_text_file requests = %+v, want none despite the capability", writes)
	}
	reqBodies := requests()
	if len(reqBodies) != 2 {
		t.Fatalf("model requests = %d, want 2 (tool call round)", len(reqBodies))
	}
	if !strings.Contains(reqBodies[1], "content from the agent's disk") {
		t.Errorf("second request missing the disk contents:\n%s", reqBodies[1])
	}
	if strings.Contains(reqBodies[1], "file body from client") {
		t.Errorf("second request carries the client's answer, want the agent's disk:\n%s", reqBodies[1])
	}
}

// TestEditFileRunsAfterApprovedRead drives the flow the memoized middleware
// exists for: the model reads, the gate holds it, the read runs on approval;
// the model then edits, the gate holds that too, and the edit must still find
// the file-state the read recorded — which lived in an earlier Generate of
// this turn. Without the per-turn instance the resumed call would start with
// an empty cache and refuse the edit as unread.
func TestEditFileRunsAfterApprovedRead(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cwd := t.TempDir()
	wantPath := filepath.Join(cwd, "hello.txt")
	seedFile(t, wantPath, "hello world\n")

	requests := startToolModelCalls(t, func(n int) (string, string) {
		switch n {
		case 1:
			return "read_file", `{"filePath":"hello.txt"}`
		case 2:
			return "edit_file", `{"filePath":"hello.txt","edits":[{"oldString":"hello world","newString":"hello there"}]}`
		}
		return "", ""
	}, "the file is edited")

	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))

	_, clientConn, agentConn, rc := openAgentPipes(t, g, fakeModel())

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	sess, err := clientConn.NewSession(ctx, acp.NewSessionRequest{Cwd: cwd, McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	if _, err := clientConn.Prompt(ctx, acp.PromptRequest{
		SessionId: sess.SessionId,
		Prompt:    []acp.ContentBlock{acp.TextBlock("change the greeting")},
	}); err != nil {
		t.Fatalf("prompt: %v", err)
	}
	_ = agentConn.Done()
	_ = clientConn.Done()

	// One prompt per tool: the gate held both, each ran after its own
	// allow-once answer.
	if perms := rc.permissionRequests(); len(perms) != 2 {
		t.Fatalf("permission requests = %d (%+v), want one per tool", len(perms), perms)
	}
	data, err := os.ReadFile(wantPath)
	if err != nil {
		t.Fatalf("read edited file: %v", err)
	}
	if string(data) != "hello there\n" {
		t.Errorf("file content = %q, want the edit applied", data)
	}
	if writes := rc.fsWrites(); len(writes) != 0 {
		t.Errorf("fs/write_text_file requests = %+v, want none", writes)
	}
	if bodies := requests(); len(bodies) != 3 {
		t.Fatalf("model requests = %d, want 3 (read round, edit round, answer)", len(bodies))
	}
}

// TestEditFileRefusesUnreadFile checks the middleware's own guard from the
// model's side: editing a file this turn never read fails, the file is left
// alone, and the failure reaches the model as something it can act on.
func TestEditFileRefusesUnreadFile(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cwd := t.TempDir()
	wantPath := filepath.Join(cwd, "hello.txt")
	seedFile(t, wantPath, "hello world\n")

	requests := startToolModel(t, "edit_file",
		`{"filePath":"hello.txt","edits":[{"oldString":"hello world","newString":"hello there"}]}`,
		"the edit is refused")

	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))

	_, clientConn, agentConn, _ := openAgentPipes(t, g, fakeModel())

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	sess, err := clientConn.NewSession(ctx, acp.NewSessionRequest{Cwd: cwd, McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	if _, err := clientConn.Prompt(ctx, acp.PromptRequest{
		SessionId: sess.SessionId,
		Prompt:    []acp.ContentBlock{acp.TextBlock("change the greeting")},
	}); err != nil {
		t.Fatalf("prompt: %v", err)
	}
	_ = agentConn.Done()
	_ = clientConn.Done()

	data, err := os.ReadFile(wantPath)
	if err != nil {
		t.Fatalf("read file: %v", err)
	}
	if string(data) != "hello world\n" {
		t.Errorf("file content = %q, want it untouched", data)
	}
	bodies := requests()
	if len(bodies) != 2 {
		t.Fatalf("model requests = %d, want 2", len(bodies))
	}
	if !strings.Contains(bodies[1], "has not been read yet") {
		t.Errorf("second request missing the refusal the model needs:\n%s", bodies[1])
	}
}

// TestListFilesToolListsRoot covers the tool that is always registered: one
// listing of the workspace root comes back with what is under it, over the
// same pipes and through the same approval as the rest.
func TestListFilesToolListsRoot(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cwd := t.TempDir()
	seedFile(t, filepath.Join(cwd, "a.txt"), "a\n")
	if err := os.MkdirAll(filepath.Join(cwd, "sub"), 0o755); err != nil {
		t.Fatalf("mkdir sub: %v", err)
	}

	requests := startToolModel(t, "list_files", `{}`, "the files are listed")

	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))

	_, clientConn, agentConn, _ := openAgentPipes(t, g, fakeModel())

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	sess, err := clientConn.NewSession(ctx, acp.NewSessionRequest{Cwd: cwd, McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	if _, err := clientConn.Prompt(ctx, acp.PromptRequest{
		SessionId: sess.SessionId,
		Prompt:    []acp.ContentBlock{acp.TextBlock("what is in here?")},
	}); err != nil {
		t.Fatalf("prompt: %v", err)
	}
	_ = agentConn.Done()
	_ = clientConn.Done()

	bodies := requests()
	if len(bodies) != 2 {
		t.Fatalf("model requests = %d, want 2 (tool call round)", len(bodies))
	}
	for _, want := range []string{"a.txt", "sub"} {
		if !strings.Contains(bodies[1], want) {
			t.Errorf("second request missing %q:\n%s", want, bodies[1])
		}
	}
}

// TestFileToolsConfinedToRoot checks the boundary os.Root enforces: a path
// climbing out of the session cwd fails, the file outside it never reaches
// the model, and the failure comes back as something the model reads.
func TestFileToolsConfinedToRoot(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	outer := t.TempDir()
	cwd := filepath.Join(outer, "ws")
	if err := os.MkdirAll(cwd, 0o755); err != nil {
		t.Fatalf("mkdir workspace: %v", err)
	}
	secret := "TOP SECRET OUTSIDE THE ROOT\n"
	if err := os.WriteFile(filepath.Join(outer, "secret.txt"), []byte(secret), 0o644); err != nil {
		t.Fatalf("write secret: %v", err)
	}

	requests := startToolModel(t, "read_file", `{"filePath":"../secret.txt"}`, "never reached")

	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))

	_, clientConn, agentConn, _ := openAgentPipes(t, g, fakeModel())

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	sess, err := clientConn.NewSession(ctx, acp.NewSessionRequest{Cwd: cwd, McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	if _, err := clientConn.Prompt(ctx, acp.PromptRequest{
		SessionId: sess.SessionId,
		Prompt:    []acp.ContentBlock{acp.TextBlock("read the secret")},
	}); err != nil {
		t.Fatalf("prompt: %v", err)
	}
	_ = agentConn.Done()
	_ = clientConn.Done()

	bodies := requests()
	if len(bodies) != 2 {
		t.Fatalf("model requests = %d, want 2 (the refused read still answers the call)", len(bodies))
	}
	if strings.Contains(bodies[1], "TOP SECRET") {
		t.Errorf("second request carries content from outside the root:\n%s", bodies[1])
	}
	if !strings.Contains(bodies[1], "failed") {
		t.Errorf("second request missing the failure the model reads:\n%s", bodies[1])
	}
}
