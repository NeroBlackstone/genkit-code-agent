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
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/deepseek"
)

// TestWriteToolRunsInTurn drives a whole tool round through the real ACP
// pipes: the fake model asks for a write on the first request, genkit runs
// the tool, and the second request carries the tool result that lets the
// model answer. The file on disk and the stored transcript are what the
// round must leave behind.
func TestWriteToolRunsInTurn(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	cwd := t.TempDir()

	var (
		mu        sync.Mutex
		reqCount  int
		reqBodies []string
	)

	toolCallArg := fmt.Sprintf(`{"filePath":%q,"content":"hello from tool"}`, filepath.Join(cwd, "sub", "out.txt"))

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		reqCount++
		n := reqCount
		reqBodies = append(reqBodies, string(body))
		mu.Unlock()

		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"object":"list","data":[]}`)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		chunks := firstToolCallChunks(toolCallArg)
		if n > 1 {
			chunks = finalAnswerChunks("the file is written")
		}
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

	ag, clientConn, agentConn, _ := openAgentPipes(t, g, fakeModel())

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

	// The tool ran: the file exists with the content the model asked for,
	// parent directories included.
	written, err := os.ReadFile(filepath.Join(cwd, "sub", "out.txt"))
	if err != nil {
		t.Fatalf("read written file: %v", err)
	}
	if string(written) != "hello from tool" {
		t.Errorf("file content = %q, want %q", written, "hello from tool")
	}

	mu.Lock()
	defer mu.Unlock()
	if len(reqBodies) != 2 {
		t.Fatalf("model requests = %d, want 2 (tool call round)", len(reqBodies))
	}
	// The second request must carry the tool response so the model can see
	// what its call did.
	if !strings.Contains(reqBodies[1], "Wrote 15 bytes to") {
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

// firstToolCallChunks streams a single tool call asking for the write tool.
func firstToolCallChunks(args string) []string {
	return []string{
		fmt.Sprintf(`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"write","arguments":%q}}]},"finish_reason":null}]}`, args),
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

// TestRunWriteResolvesRelativePath checks the tool's own path rules: a
// relative filePath lands under the session cwd carried in the context, an
// absolute one is used as-is, and both create missing parent directories.
func TestRunWriteResolvesRelativePath(t *testing.T) {
	dir := t.TempDir()
	tool := defineTools(genkit.Init(context.Background()))
	if len(tool) != 1 || tool[0].Name() != "write" {
		t.Fatalf("tools = %v, want the write tool", tool)
	}

	ctx := core.WithActionContext(context.Background(), core.ActionContext{"cwd": dir})
	tw := tool[0].(ai.Tool)
	out, err := tw.RunRaw(ctx, WriteInput{FilePath: "a/b.txt", Content: "x"})
	if err != nil {
		t.Fatalf("run write: %v", err)
	}
	if s, _ := out.(string); !strings.Contains(s, "Wrote 1 bytes to") {
		t.Errorf("output = %v, want a byte count", out)
	}
	got, err := os.ReadFile(filepath.Join(dir, "a", "b.txt"))
	if err != nil || string(got) != "x" {
		t.Errorf("file = %q, err = %v, want %q", got, err, "x")
	}

	// Without a cwd in the context the process directory decides, which for
	// an absolute path means nothing: it is written where it says.
	abs := filepath.Join(dir, "c.txt")
	if _, err := tw.RunRaw(context.Background(), WriteInput{FilePath: abs, Content: "y"}); err != nil {
		t.Fatalf("run write absolute: %v", err)
	}
	if got, err := os.ReadFile(abs); err != nil || string(got) != "y" {
		t.Errorf("file = %q, err = %v, want %q", got, err, "y")
	}
}

// TestWithCwdUsesActionContext pins the doc-aligned channel: withCwd writes
// genkit's action context, keeps keys an earlier caller set, and an empty
// session cwd leaves the context alone.
func TestWithCwdUsesActionContext(t *testing.T) {
	base := core.WithActionContext(context.Background(), core.ActionContext{"uid": "alice"})
	ctx := withCwd(base, "/session/dir")

	if got := cwdFrom(ctx); got != "/session/dir" {
		t.Errorf("cwdFrom = %q, want the session cwd", got)
	}
	if uid, _ := core.FromContext(ctx)["uid"].(string); uid != "alice" {
		t.Errorf("action context uid = %q, want it preserved", uid)
	}

	if got := withCwd(base, ""); got != base {
		t.Error("withCwd with an empty cwd changed the context")
	}
}
