package agent

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

// TestTurnFilesystemBuildsOncePerTurn pins the memoization: every New the
// approval loop's resumes ask for gets the same hook bundle back, so the
// file-state cache one Generate records is the one the next one reads, and
// the bundle carries the four file tools the middleware registers.
func TestTurnFilesystemBuildsOncePerTurn(t *testing.T) {
	f := newTurnFilesystem(t.TempDir())

	first, err := f.New(context.Background())
	if err != nil {
		t.Fatalf("first new: %v", err)
	}
	second, err := f.New(context.Background())
	if err != nil {
		t.Fatalf("second new: %v", err)
	}
	if first != second {
		t.Error("New built a second bundle, want the turn's first one reused")
	}
	if first.WrapGenerate == nil {
		t.Fatal("hooks carry no WrapGenerate")
	}

	names := make([]string, 0, len(first.Tools))
	for _, tool := range first.Tools {
		names = append(names, tool.Name())
	}
	for _, want := range []string{"list_files", "read_file", "write_file", "edit_file"} {
		found := false
		for _, name := range names {
			if name == want {
				found = true
			}
		}
		if !found {
			t.Errorf("tools = %v, want %q among them", names, want)
		}
	}
}

// recordNext builds a WrapGenerate continuation that captures the messages
// it is handed, which is how these tests see what survived a hook.
func recordNext(into *[]*ai.Message) ai.GenerateNext {
	return func(_ context.Context, p *ai.GenerateParams) (*ai.ModelResponse, error) {
		*into = append([]*ai.Message(nil), p.Request.Messages...)
		return &ai.ModelResponse{}, nil
	}
}

// readForTest runs the bundle's read_file once so its injection is queued,
// standing in for an approved read_file earlier in the turn.
func readForTest(t *testing.T, hooks *ai.Hooks, path string) {
	t.Helper()
	for _, tool := range hooks.Tools {
		if tool.Name() != "read_file" {
			continue
		}
		if _, err := tool.RunRaw(context.Background(), map[string]any{"filePath": path}); err != nil {
			t.Fatalf("run read_file: %v", err)
		}
		return
	}
	t.Fatal("hooks carry no read_file tool")
}

// countInjected returns how many messages in the conversation carry the
// read_file envelope the middleware injects file contents inside.
func countInjected(msgs []*ai.Message) int {
	n := 0
	for _, m := range msgs {
		if m != nil && strings.Contains(m.Text(), "<read_file path=") {
			n++
		}
	}
	return n
}

// testResumeOptions builds the options a resumed Generate carries: a call
// with one restart for the gate to resolve on its first iteration.
func testResumeOptions() *ai.GenerateActionOptions {
	return &ai.GenerateActionOptions{Resume: &ai.GenerateActionResume{
		Restart: []*ai.Part{ai.NewToolRequestPart(&ai.ToolRequest{Name: "edit_file", Ref: "call_1"})},
	}}
}

// resumeParams builds the shape genkit hands WrapGenerate on the iteration
// that resolves resume directives: iteration 0 of a call carrying restarts,
// whose request genkit rebuilds from the Generate's own options after the
// hooks have run.
func resumeParams(messages []*ai.Message) *ai.GenerateParams {
	return &ai.GenerateParams{
		Request:   &ai.ModelRequest{Messages: messages},
		Options:   testResumeOptions(),
		Iteration: 0,
	}
}

// TestTurnFilesystemRequeuesDroppedInjections covers the bookkeeping that
// keeps read_file's injected contents from vanishing: the turn-0 resume
// iteration is one genkit rebuilds its request on, dropping whatever the
// hooks appended, so the middleware's injection is captured on the way
// through and re-delivered at the next iteration — the one the model is
// actually called on — exactly once.
func TestTurnFilesystemRequeuesDroppedInjections(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("hello from disk"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}
	f := newTurnFilesystem(dir)
	hooks, err := f.New(context.Background())
	if err != nil {
		t.Fatalf("new hooks: %v", err)
	}
	readForTest(t, hooks, "a.txt")

	prompt := []*ai.Message{ai.NewUserTextMessage("look at it")}

	var droppedSeen []*ai.Message
	if _, err := hooks.WrapGenerate(context.Background(), resumeParams(prompt), recordNext(&droppedSeen)); err != nil {
		t.Fatalf("wrap generate on the resume iteration: %v", err)
	}
	// The middleware injected into this iteration's request, and the
	// capture held a copy back for the iteration that survives.
	if countInjected(droppedSeen) != 1 {
		t.Fatalf("resume iteration saw %d injections, want 1", countInjected(droppedSeen))
	}
	if got := len(f.pending); got != 1 {
		t.Fatalf("pending = %d messages, want the dropped injection held", got)
	}

	// The next iteration: genkit's rebuilt request never saw the
	// injection, so the held copy must be re-added there.
	next := &ai.GenerateParams{
		Request:   &ai.ModelRequest{Messages: append([]*ai.Message(nil), prompt...)},
		Options:   testResumeOptions(),
		Iteration: 1,
	}
	var survived []*ai.Message
	if _, err := hooks.WrapGenerate(context.Background(), next, recordNext(&survived)); err != nil {
		t.Fatalf("wrap generate on the next iteration: %v", err)
	}
	if countInjected(survived) != 1 {
		t.Fatalf("surviving iteration saw %d injections, want exactly 1", countInjected(survived))
	}
	if len(survived) != len(prompt)+1 {
		t.Errorf("surviving messages = %d, want the prompt plus one injection", len(survived))
	}
	if got := len(f.pending); got != 0 {
		t.Errorf("pending = %d messages after the flush, want it emptied", got)
	}
}

// TestTurnFilesystemKeepsSurvivingInjection is the control for the requeue:
// on an iteration whose request genkit does NOT rebuild — every non-resume
// iteration — the middleware's injection stays where it was put, and the
// bookkeeping must not queue a copy that would later double it.
func TestTurnFilesystemKeepsSurvivingInjection(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("hello from disk"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}
	f := newTurnFilesystem(dir)
	hooks, err := f.New(context.Background())
	if err != nil {
		t.Fatalf("new hooks: %v", err)
	}
	readForTest(t, hooks, "a.txt")

	prompt := []*ai.Message{ai.NewUserTextMessage("look at it")}
	fresh := &ai.GenerateParams{
		Request:   &ai.ModelRequest{Messages: prompt},
		Options:   &ai.GenerateActionOptions{},
		Iteration: 0,
	}
	var delivered []*ai.Message
	if _, err := hooks.WrapGenerate(context.Background(), fresh, recordNext(&delivered)); err != nil {
		t.Fatalf("wrap generate: %v", err)
	}
	if countInjected(delivered) != 1 {
		t.Fatalf("iteration saw %d injections, want 1", countInjected(delivered))
	}
	if got := len(f.pending); got != 0 {
		t.Errorf("pending = %d messages, want none held on a request that survives", got)
	}
}
