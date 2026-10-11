package agent

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/deepseek"
)

// writeSkill installs one skill under root/skills/<dir>: a SKILL.md fenced by
// the given YAML frontmatter, carrying body after it. dir may carry a path
// separator to plant a SKILL.md below a direct subdirectory, which discovery
// must not reach.
func writeSkill(t *testing.T, root, dir, frontmatter, body string) {
	t.Helper()
	skillDir := filepath.Join(root, "skills", dir)
	if err := os.MkdirAll(skillDir, 0o755); err != nil {
		t.Fatalf("mkdir skill dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(skillDir, "SKILL.md"), []byte("---\n"+frontmatter+"---\n"+body), 0o644); err != nil {
		t.Fatalf("write SKILL.md: %v", err)
	}
}

// promptOnce drives one prompt through the real ACP pipes against the fake
// model the test already started — before genkit.Init, since the deepseek
// plugin reads DEEPSEEK_BASE_URL at init — and returns the requests it
// produced. The working directory is the process one for the duration, which
// is where the skills middleware's default "skills" path resolves against.
func promptOnce(t *testing.T, g *genkit.Genkit, fake *fakeDeepSeek, prompt string) func() []string {
	t.Helper()

	_, clientConn, agentConn, _ := openAgentPipes(t, g, fakeModel())

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	sess, err := clientConn.NewSession(ctx, acp.NewSessionRequest{Cwd: t.TempDir(), McpServers: []acp.McpServer{}})
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
	return fake.requests
}

// listsSkills reports whether a recorded request body carries the injected
// skills block: the body is JSON, where "<" arrives escaped as <.
func listsSkills(body string) bool {
	return strings.Contains(body, "Here are the available skills") ||
		strings.Contains(body, "<skills>")
}

// TestSkillsListReachesModel checks the hot path: only a skill's directory
// name and its frontmatter description reach the prompt, the body stays
// behind use_skill, and what discovery deliberately hides — a nested
// directory, a dotted one, and the frontmatter's name field — is not among
// them.
func TestSkillsListReachesModel(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	root := t.TempDir()
	writeSkill(t, root, "doorkeeper",
		"name: ignored-name\ndescription: Answers only through the door.\n",
		"Only speak through the door. Never speak directly.\n")
	// A SKILL.md below a direct subdirectory is out of reach: discovery
	// looks at direct children only.
	writeSkill(t, root, filepath.Join("nested", "deep"), "", "buried body\n")
	writeSkill(t, root, ".secret", "description: Hidden from the list.\n", "secret body\n")
	t.Chdir(root)

	fake := startFakeDeepSeek(t)
	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))
	requests := promptOnce(t, g, fake, "how do I get through?")

	bodies := requests()
	if len(bodies) == 0 {
		t.Fatal("no model request reached the endpoint")
	}
	body := bodies[0]
	for _, want := range []string{"doorkeeper", "Answers only through the door.", useSkillToolName} {
		if !strings.Contains(body, want) {
			t.Errorf("first request missing %q:\n%s", want, body)
		}
	}
	if !listsSkills(body) {
		t.Errorf("first request missing the skills list:\n%s", body)
	}
	// The list carries names and descriptions, nothing heavier: no body,
	// no frontmatter name, nothing a scan rule hides.
	for _, unwanted := range []string{"ignored-name", "buried body", "secret body", "Hidden from the list", "Only speak through the door"} {
		if strings.Contains(body, unwanted) {
			t.Errorf("first request carries %q, want it kept behind use_skill:\n%s", unwanted, body)
		}
	}
}

// TestWithoutSkillsDirectorySkipsInjection pins the zero-config default: a
// missing "skills" directory under the working directory is not an error —
// the turn runs and the list is simply absent.
func TestWithoutSkillsDirectorySkipsInjection(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Chdir(t.TempDir())

	fake := startFakeDeepSeek(t)
	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))
	requests := promptOnce(t, g, fake, "hello")

	bodies := requests()
	if len(bodies) == 0 {
		t.Fatal("no model request reached the endpoint")
	}
	if listsSkills(bodies[0]) {
		t.Errorf("request carries a skills list with no skills directory:\n%s", bodies[0])
	}
}

// TestUseSkillLoadsSkillWithoutPermission drives the load the prompt asks
// for: use_skill runs with no session/request_permission, since it is
// pre-approved; it is still reported to the client as a tool call, titled
// after the skill it names; its SKILL.md body reaches the model's next
// request; and a second prompt replays the load from the stored history
// with the skills list freshly injected alongside it.
func TestUseSkillLoadsSkillWithoutPermission(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	root := t.TempDir()
	writeSkill(t, root, "doorkeeper", "description: Answers only through the door.\n",
		"Only speak through the door.\n")
	t.Chdir(root)

	requests := startToolModel(t, useSkillToolName, `{"skillName":"doorkeeper"}`, "the door answers")

	g := genkit.Init(context.Background(), genkit.WithPlugins(&deepseek.DeepSeek{APIKey: "test-key"}))
	_, clientConn, agentConn, rc := openAgentPipes(t, g, fakeModel())

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	sess, err := clientConn.NewSession(ctx, acp.NewSessionRequest{Cwd: t.TempDir(), McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	if _, err := clientConn.Prompt(ctx, acp.PromptRequest{
		SessionId: sess.SessionId,
		Prompt:    []acp.ContentBlock{acp.TextBlock("how do I get through?")},
	}); err != nil {
		t.Fatalf("first prompt: %v", err)
	}

	// The load never reached the client as a permission prompt: use_skill
	// is on the gate's allowed list.
	if perms := rc.permissionRequests(); len(perms) != 0 {
		t.Fatalf("permission requests = %d (%+v), want none: use_skill is pre-approved", len(perms), perms)
	}
	// It was reported like any other tool call, titled after the skill.
	events := waitForToolEvents(t, rc, 3)
	create := events[0].ToolCall
	if create == nil || create.Status != acp.ToolCallStatusPending {
		t.Fatalf("first update = %+v, want the pending creating update", events[0])
	}
	if create.Title != "Loading skill doorkeeper" {
		t.Errorf("title = %q, want %q", create.Title, "Loading skill doorkeeper")
	}
	if create.Kind != acp.ToolKindOther {
		t.Errorf("kind = %q, want %q", create.Kind, acp.ToolKindOther)
	}
	if s := statusOf(events[1]); s != acp.ToolCallStatusInProgress {
		t.Errorf("second status = %q, want in_progress", s)
	}
	update := events[2].ToolCallUpdate
	if update == nil || update.Status == nil || *update.Status != acp.ToolCallStatusCompleted {
		t.Fatalf("third update = %+v, want the completed tool_call_update", events[2])
	}
	if got := fmt.Sprintf("%v", update.RawOutput); !strings.Contains(got, "Only speak through the door.") {
		t.Errorf("rawOutput = %q, want the skill body", got)
	}

	bodies := requests()
	if len(bodies) != 2 {
		t.Fatalf("model requests = %d, want 2 (tool call round)", len(bodies))
	}
	if !listsSkills(bodies[0]) {
		t.Errorf("first request missing the skills list:\n%s", bodies[0])
	}
	if !strings.Contains(bodies[1], "Only speak through the door.") {
		t.Errorf("second request missing the skill body:\n%s", bodies[1])
	}

	// A second prompt replays the load: the body rides in the stored
	// history, and the list is injected again for the new turn.
	if _, err := clientConn.Prompt(ctx, acp.PromptRequest{
		SessionId: sess.SessionId,
		Prompt:    []acp.ContentBlock{acp.TextBlock("and now?")},
	}); err != nil {
		t.Fatalf("second prompt: %v", err)
	}
	_ = agentConn.Done()
	_ = clientConn.Done()

	bodies = requests()
	if len(bodies) != 3 {
		t.Fatalf("model requests = %d, want 3 (the second prompt answers directly)", len(bodies))
	}
	if !strings.Contains(bodies[2], "Only speak through the door.") {
		t.Errorf("second prompt's request missing the stored skill body:\n%s", bodies[2])
	}
	if !listsSkills(bodies[2]) {
		t.Errorf("second prompt's request missing a freshly injected skills list:\n%s", bodies[2])
	}
}
