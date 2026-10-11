package agent

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/deepseek"
)

// TestWriteToolRunsInTurn drives a whole tool round through the real ACP
// pipes: the fake model asks for a write on the first request, genkit runs
// the tool, and the second request carries the tool result that lets the
// model answer. The client advertised no fs capability, so the tool must
// refuse with a message the model can read: the protocol forbids calling
// fs/write_text_file, and nothing writes the file in its place. The stored
// transcript is what the round must leave behind either way.
func TestWriteToolRunsInTurn(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cwd := t.TempDir()

	toolCallArg := fmt.Sprintf(`{"filePath":%q,"content":"hello from tool"}`, filepath.Join(cwd, "sub", "out.txt"))
	requests := startToolModel(t, "write", toolCallArg, "the file is written")

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

	// The tool had nowhere to write: no fs/write_text_file was called, and
	// nothing landed on the agent's own disk in its place.
	if writes := rc.fsWrites(); len(writes) != 0 {
		t.Errorf("fs/write_text_file requests = %+v, want none without the capability", writes)
	}
	if _, err := os.Stat(filepath.Join(cwd, "sub", "out.txt")); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("stat requested file = %v, want it never written locally", err)
	}

	reqBodies := requests()
	if len(reqBodies) != 2 {
		t.Fatalf("model requests = %d, want 2 (tool call round)", len(reqBodies))
	}
	// The second request must carry the tool response so the model can see
	// what its call did: the refusal, naming the file it could not write.
	if !strings.Contains(reqBodies[1], "the client does not support fs/write_text_file") {
		t.Errorf("second request missing the refusal:\n%s", reqBodies[1])
	}
	if !strings.Contains(reqBodies[1], filepath.Join(cwd, "sub", "out.txt")) {
		t.Errorf("second request's tool result missing the path:\n%s", reqBodies[1])
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

// TestWriteToolGoesOverACP drives the same tool round against a client that
// advertised fs.writeTextFile: the write must reach the client as one
// fs/write_text_file request, carrying the session id and an absolute path,
// and the agent's own disk must stay untouched, since the file now belongs
// to the client and writing it twice is the bug this path exists to avoid.
func TestWriteToolGoesOverACP(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cwd := t.TempDir()
	wantPath := filepath.Join(cwd, "sub", "out.txt")

	toolCallArg := fmt.Sprintf(`{"filePath":%q,"content":"hello from tool"}`, wantPath)
	requests := startToolModel(t, "write", toolCallArg, "the file is written")

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
		Prompt:    []acp.ContentBlock{acp.TextBlock("write hello to out.txt")},
	}); err != nil {
		t.Fatalf("prompt: %v", err)
	}
	_ = agentConn.Done()
	_ = clientConn.Done()

	writes := rc.fsWrites()
	if len(writes) != 1 {
		t.Fatalf("fs/write_text_file requests = %+v, want exactly one", writes)
	}
	w := writes[0]
	if w.SessionId != sess.SessionId {
		t.Errorf("sessionId = %q, want the session the turn ran on %q", w.SessionId, sess.SessionId)
	}
	if w.Path != wantPath {
		t.Errorf("path = %q, want the absolute %q", w.Path, wantPath)
	}
	if w.Content != "hello from tool" {
		t.Errorf("content = %q, want what the model asked for", w.Content)
	}
	// The agent wrote nothing itself: the client owns the file now.
	if _, err := os.Stat(wantPath); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("stat %s = %v, want the file left to the client", wantPath, err)
	}

	reqBodies := requests()
	if len(reqBodies) != 2 {
		t.Fatalf("model requests = %d, want 2 (tool call round)", len(reqBodies))
	}
	if !strings.Contains(reqBodies[1], "Wrote 15 bytes to") {
		t.Errorf("second request missing the tool result:\n%s", reqBodies[1])
	}
}

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

// startToolModel serves the fake model the tool tests drive: the first
// request answers with a tool call on toolName carrying toolCallArg, every
// later one with finalText. It returns the request bodies it recorded, in
// arrival order, and points DEEPSEEK_BASE_URL at itself for the duration of
// the test.
func startToolModel(t *testing.T, toolName, toolCallArg, finalText string) func() []string {
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
		chunks := firstToolCallChunks(toolName, toolCallArg)
		if n > 1 {
			chunks = finalAnswerChunks(finalText)
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

// mustTool returns the named tool among refs, failing the test when the
// registration did not produce one.
func mustTool(t *testing.T, refs []ai.ToolRef, name string) ai.Tool {
	t.Helper()
	for _, ref := range refs {
		if ref.Name() == name {
			return ref.(ai.Tool)
		}
	}
	t.Fatalf("registered tools = %v, want %q among them", refs, name)
	return nil
}

// TestRunWriteResolvesRelativePath checks the tool's own path rules: a
// relative filePath lands under the session cwd carried in the context, an
// absolute one is used as-is, and both reach the turn's writer as absolute
// paths, the only kind fs/write_text_file takes. With no writer at all the
// tool refuses, since writing the file somewhere the client cannot see is
// the bug the protocol exists to prevent.
func TestRunWriteResolvesRelativePath(t *testing.T) {
	dir := t.TempDir()
	tw := mustTool(t, defineTools(genkit.Init(context.Background())), "write")

	type writeRecord struct{ path, content string }
	var writes []writeRecord
	ctx := withToolEnv(context.Background(), toolEnv{
		cwd: dir,
		write: func(_ context.Context, path, content string) error {
			writes = append(writes, writeRecord{path, content})
			return nil
		},
	})

	out, err := tw.RunRaw(ctx, WriteInput{FilePath: "a/b.txt", Content: "x"})
	if err != nil {
		t.Fatalf("run write: %v", err)
	}
	if s, _ := out.(string); !strings.Contains(s, "Wrote 1 bytes to") {
		t.Errorf("output = %v, want a byte count", out)
	}

	abs := filepath.Join(dir, "c.txt")
	if _, err := tw.RunRaw(ctx, WriteInput{FilePath: abs, Content: "y"}); err != nil {
		t.Fatalf("run write absolute: %v", err)
	}

	want := []writeRecord{
		{filepath.Join(dir, "a", "b.txt"), "x"},
		{abs, "y"},
	}
	if len(writes) != len(want) {
		t.Fatalf("writes = %+v, want %+v", writes, want)
	}
	for i := range want {
		if writes[i] != want[i] {
			t.Errorf("write %d = %+v, want %+v", i, writes[i], want[i])
		}
	}
	// The writer is the only one that receives the file: the agent's own
	// disk stays untouched.
	for _, p := range []string{want[0].path, abs} {
		if _, err := os.Stat(p); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("stat %s = %v, want the file written only through the client", p, err)
		}
	}

	// No turn, no writer: the tool reports that to the model instead of
	// falling back to a write the client would never learn about.
	out, err = tw.RunRaw(context.Background(), WriteInput{FilePath: filepath.Join(dir, "d.txt"), Content: "z"})
	if err != nil {
		t.Fatalf("run write without a writer: %v", err)
	}
	if s, _ := out.(string); !strings.Contains(s, "the client does not support fs/write_text_file") {
		t.Errorf("output = %v, want the refusal naming the missing capability", out)
	}
	if _, err := os.Stat(filepath.Join(dir, "d.txt")); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("stat refused file = %v, want it never written", err)
	}
}

// TestToolEnvUsesActionContext pins the doc-aligned channel: withToolEnv
// writes genkit's action context, keeps the keys an earlier caller set, and
// the whole environment reads back out of it. A context no turn touched
// carries no environment, whose writer is nil, leaving tools nowhere to write.
func TestToolEnvUsesActionContext(t *testing.T) {
	base := core.WithActionContext(context.Background(), core.ActionContext{"uid": "alice"})
	stub := func(context.Context, string, string) error { return nil }
	readStub := func(context.Context, string, *int, *int) (string, error) { return "", nil }
	ctx := withToolEnv(base, toolEnv{cwd: "/session/dir", read: readStub, write: stub})

	env := envFrom(ctx)
	if env.cwd != "/session/dir" {
		t.Errorf("env.cwd = %q, want the session cwd", env.cwd)
	}
	if env.read == nil {
		t.Error("env.read = nil, want the reader the turn picked")
	}
	if env.write == nil {
		t.Error("env.write = nil, want the writer the turn picked")
	}
	if uid, _ := core.FromContext(ctx)["uid"].(string); uid != "alice" {
		t.Errorf("action context uid = %q, want it preserved", uid)
	}

	if env := envFrom(context.Background()); env.read != nil || env.write != nil || env.cwd != "" {
		t.Errorf("env without a turn = %+v, want the zero environment", env)
	}
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	if got := cwdFrom(context.Background()); got != wd {
		t.Errorf("cwdFrom = %q, want the process directory %q", got, wd)
	}
}

// TestInitializeRecordsClientCapabilities pins the gate writerFor and
// readerFor read: what the client advertised in initialize, and nothing
// before it.
func TestInitializeRecordsClientCapabilities(t *testing.T) {
	ag := New(nil, nil)
	if ag.fsWriteCapable() {
		t.Fatal("fsWriteCapable before initialize = true, want false")
	}
	if ag.fsReadCapable() {
		t.Fatal("fsReadCapable before initialize = true, want false")
	}
	if _, err := ag.Initialize(context.Background(), acp.InitializeRequest{
		ProtocolVersion: acp.ProtocolVersionNumber,
		ClientCapabilities: acp.ClientCapabilities{
			Fs: acp.FileSystemCapabilities{ReadTextFile: true, WriteTextFile: true},
		},
	}); err != nil {
		t.Fatalf("initialize: %v", err)
	}
	if !ag.fsWriteCapable() {
		t.Error("fsWriteCapable after initialize = false, want the advertised capability")
	}
	if !ag.fsReadCapable() {
		t.Error("fsReadCapable after initialize = false, want the advertised capability")
	}
}

// TestWriterForNeedsClientSupport covers the writer selection the pipes do
// not show on their own: without a connection, writerFor hands out no writer
// even when the capability was advertised, and without the capability it
// never gets far enough to send anything. Nothing writes the file in its
// place.
func TestWriterForNeedsClientSupport(t *testing.T) {
	capable := New(nil, nil)
	if _, err := capable.Initialize(context.Background(), acp.InitializeRequest{
		ProtocolVersion: acp.ProtocolVersionNumber,
		ClientCapabilities: acp.ClientCapabilities{
			Fs: acp.FileSystemCapabilities{WriteTextFile: true},
		},
	}); err != nil {
		t.Fatalf("initialize: %v", err)
	}

	for _, tc := range []struct {
		name string
		ag   *Agent
	}{
		{"never initialized", New(nil, nil)},
		{"capability without a connection", capable},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if w := tc.ag.writerFor("sess_x"); w != nil {
				t.Error("writerFor = non-nil, want nil when fs/write_text_file may not be called")
			}
		})
	}
}

// TestReadToolGoesOverACP drives a read round against a client that
// advertised fs.readTextFile: the model's relative path must reach the
// client as one fs/read_text_file request carrying the session id and the
// absolute path under the session cwd, and the content the model sees must
// be the client's answer, since the client is the one holding the file,
// unsaved editor changes included.
func TestReadToolGoesOverACP(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cwd := t.TempDir()
	wantPath := filepath.Join(cwd, "src", "main.go")

	toolCallArg := fmt.Sprintf(`{"filePath":%q}`, filepath.Join("src", "main.go"))
	requests := startToolModel(t, "read", toolCallArg, "the file is read")

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

	reads := rc.fsReads()
	if len(reads) != 1 {
		t.Fatalf("fs/read_text_file requests = %+v, want exactly one", reads)
	}
	r := reads[0]
	if r.SessionId != sess.SessionId {
		t.Errorf("sessionId = %q, want the session the turn ran on %q", r.SessionId, sess.SessionId)
	}
	if r.Path != wantPath {
		t.Errorf("path = %q, want the relative one resolved to the absolute %q", r.Path, wantPath)
	}
	if r.Line != nil || r.Limit != nil {
		t.Errorf("line/limit = %v/%v, want nil when the model omitted them", r.Line, r.Limit)
	}

	reqBodies := requests()
	if len(reqBodies) != 2 {
		t.Fatalf("model requests = %d, want 2 (tool call round)", len(reqBodies))
	}
	if !strings.Contains(reqBodies[1], "file body from client") {
		t.Errorf("second request missing the content the client returned:\n%s", reqBodies[1])
	}
}

// TestRunReadResolvesRelativePath checks the read tool's own path rules: a
// relative filePath lands under the session cwd carried in the context, an
// absolute one is used as-is, both reach the turn's reader as absolute
// paths, and the optional line window rides along untouched. With no reader
// at all the tool refuses, since reading the file off the agent's own disk
// would show the model a copy the client may already have changed.
func TestRunReadResolvesRelativePath(t *testing.T) {
	dir := t.TempDir()
	refs := defineTools(genkit.Init(context.Background()))
	if len(refs) != 2 || refs[0].Name() != "read" || refs[1].Name() != "write" {
		t.Fatalf("registered tools = %v, want read registered before write", refs)
	}
	tw := refs[0].(ai.Tool)

	toVal := func(p *int) int {
		if p == nil {
			return -1
		}
		return *p
	}
	type readRecord struct {
		path        string
		line, limit int
	}
	var reads []readRecord
	ctx := withToolEnv(context.Background(), toolEnv{
		cwd: dir,
		read: func(_ context.Context, path string, line, limit *int) (string, error) {
			reads = append(reads, readRecord{path, toVal(line), toVal(limit)})
			return "file body", nil
		},
	})

	out, err := tw.RunRaw(ctx, ReadInput{FilePath: "a/b.txt"})
	if err != nil {
		t.Fatalf("run read: %v", err)
	}
	if s, _ := out.(string); s != "file body" {
		t.Errorf("output = %q, want the content the reader returned", out)
	}

	abs := filepath.Join(dir, "c.txt")
	if _, err := tw.RunRaw(ctx, ReadInput{FilePath: abs, Line: new(3), Limit: new(5)}); err != nil {
		t.Fatalf("run read absolute: %v", err)
	}

	want := []readRecord{
		{filepath.Join(dir, "a", "b.txt"), -1, -1},
		{abs, 3, 5},
	}
	if len(reads) != len(want) {
		t.Fatalf("reads = %+v, want %+v", reads, want)
	}
	for i := range want {
		if reads[i] != want[i] {
			t.Errorf("read %d = %+v, want %+v", i, reads[i], want[i])
		}
	}

	// No turn, no reader: the tool reports that to the model instead of
	// falling back to a local read the client would never see happen.
	local := filepath.Join(dir, "local.txt")
	if err := os.WriteFile(local, []byte("on the agent's own disk"), 0o600); err != nil {
		t.Fatalf("write local file: %v", err)
	}
	out, err = tw.RunRaw(context.Background(), ReadInput{FilePath: local})
	if err != nil {
		t.Fatalf("run read without a reader: %v", err)
	}
	s, _ := out.(string)
	if !strings.Contains(s, "the client does not support fs/read_text_file") {
		t.Errorf("output = %v, want the refusal naming the missing capability", out)
	}
	if strings.Contains(s, "on the agent's own disk") {
		t.Errorf("output = %q, want no fallback to the agent's own disk", out)
	}
}

// TestReaderForNeedsClientSupport covers the reader selection the pipes do
// not show on their own: without a connection, readerFor hands out no
// reader even when the capability was advertised, and a wired connection
// without fs.readTextFile gets none either, so the gate is the advertised
// capability and not just the transport.
func TestReaderForNeedsClientSupport(t *testing.T) {
	capable := New(nil, nil)
	if _, err := capable.Initialize(context.Background(), acp.InitializeRequest{
		ProtocolVersion: acp.ProtocolVersionNumber,
		ClientCapabilities: acp.ClientCapabilities{
			Fs: acp.FileSystemCapabilities{ReadTextFile: true},
		},
	}); err != nil {
		t.Fatalf("initialize: %v", err)
	}
	writeOnly, _, _, _ := openAgentPipesCaps(t, nil, fakeModel(), acp.ClientCapabilities{
		Fs: acp.FileSystemCapabilities{WriteTextFile: true},
	})

	for _, tc := range []struct {
		name string
		ag   *Agent
	}{
		{"never initialized", New(nil, nil)},
		{"capability without a connection", capable},
		{"connection without the capability", writeOnly},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if r := tc.ag.readerFor("sess_x"); r != nil {
				t.Error("readerFor = non-nil, want nil when fs/read_text_file may not be called")
			}
		})
	}
}
