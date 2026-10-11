package agent

import (
	"context"
	"io"
	"sync"
	"testing"

	"github.com/coder/acp-go-sdk"
)

type testClient struct {
	mu      sync.Mutex
	updates []string
}

var _ acp.Client = (*testClient)(nil)

func (c *testClient) ReadTextFile(ctx context.Context, p acp.ReadTextFileRequest) (acp.ReadTextFileResponse, error) {
	return acp.ReadTextFileResponse{}, acp.NewMethodNotFound(acp.ClientMethodFsReadTextFile)
}
func (c *testClient) WriteTextFile(ctx context.Context, p acp.WriteTextFileRequest) (acp.WriteTextFileResponse, error) {
	return acp.WriteTextFileResponse{}, acp.NewMethodNotFound(acp.ClientMethodFsWriteTextFile)
}
func (c *testClient) RequestPermission(ctx context.Context, p acp.RequestPermissionRequest) (acp.RequestPermissionResponse, error) {
	return acp.RequestPermissionResponse{}, acp.NewMethodNotFound("permission/request")
}
func (c *testClient) SessionUpdate(ctx context.Context, p acp.SessionNotification) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if p.Update.AgentMessageChunk != nil && p.Update.AgentMessageChunk.Content.Text != nil {
		c.updates = append(c.updates, p.Update.AgentMessageChunk.Content.Text.Text)
	}
	return nil
}
func (c *testClient) CreateTerminal(ctx context.Context, p acp.CreateTerminalRequest) (acp.CreateTerminalResponse, error) {
	return acp.CreateTerminalResponse{}, acp.NewMethodNotFound("terminal/create")
}
func (c *testClient) KillTerminal(ctx context.Context, p acp.KillTerminalRequest) (acp.KillTerminalResponse, error) {
	return acp.KillTerminalResponse{}, acp.NewMethodNotFound("terminal/kill")
}
func (c *testClient) TerminalOutput(ctx context.Context, p acp.TerminalOutputRequest) (acp.TerminalOutputResponse, error) {
	return acp.TerminalOutputResponse{}, acp.NewMethodNotFound("terminal/output")
}
func (c *testClient) ReleaseTerminal(ctx context.Context, p acp.ReleaseTerminalRequest) (acp.ReleaseTerminalResponse, error) {
	return acp.ReleaseTerminalResponse{}, acp.NewMethodNotFound("terminal/release")
}
func (c *testClient) WaitForTerminalExit(ctx context.Context, p acp.WaitForTerminalExitRequest) (acp.WaitForTerminalExitResponse, error) {
	return acp.WaitForTerminalExitResponse{}, acp.NewMethodNotFound("terminal/waitForExit")
}

func TestAgentPromptRoundTrip(t *testing.T) {
	// Keep the session store away from the real ~/.gca.
	t.Setenv("HOME", t.TempDir())

	agentToClientR, agentToClientW := io.Pipe()
	clientToAgentR, clientToAgentW := io.Pipe()

	ag := New(nil, nil)
	agentConn := acp.NewAgentSideConnection(ag, agentToClientW, clientToAgentR)
	ag.SetAgentConnection(agentConn)

	tc := &testClient{}
	clientConn := acp.NewClientSideConnection(tc, clientToAgentW, agentToClientR)

	ctx := context.Background()

	initResp, err := clientConn.Initialize(ctx, acp.InitializeRequest{
		ProtocolVersion: acp.ProtocolVersionNumber,
		ClientInfo:      &acp.Implementation{Name: "test-client", Version: "0.0.1"},
	})
	if err != nil {
		t.Fatalf("initialize: %v", err)
	}
	if initResp.ProtocolVersion != acp.ProtocolVersionNumber {
		t.Fatalf("protocol version = %v, want %v", initResp.ProtocolVersion, acp.ProtocolVersionNumber)
	}
	if initResp.AgentInfo == nil || initResp.AgentInfo.Name != "genkit-code-agent" {
		t.Fatalf("agent info = %+v", initResp.AgentInfo)
	}
	// Sessions persist to disk, so loadSession (replay from a stored
	// transcript) and resume (continue without replay) are both served.
	if initResp.AgentCapabilities.SessionCapabilities.Resume == nil {
		t.Fatal("resume capability not advertised")
	}
	if !initResp.AgentCapabilities.LoadSession {
		t.Fatal("loadSession not advertised despite stored transcripts")
	}

	sessResp, err := clientConn.NewSession(ctx, acp.NewSessionRequest{Cwd: "/tmp", McpServers: []acp.McpServer{}})
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	if sessResp.SessionId == "" {
		t.Fatal("empty session id")
	}

	promptResp, err := clientConn.Prompt(ctx, acp.PromptRequest{
		SessionId: sessResp.SessionId,
		Prompt:    []acp.ContentBlock{acp.TextBlock("hello world")},
	})
	if err != nil {
		t.Fatalf("prompt: %v", err)
	}
	if promptResp.StopReason != acp.StopReasonEndTurn {
		t.Fatalf("stop reason = %v, want %v", promptResp.StopReason, acp.StopReasonEndTurn)
	}
	// The client named no id, so the agent assigned one: the echo must be
	// present even on the modelless fallback path.
	if promptResp.UserMessageId == nil || *promptResp.UserMessageId == "" {
		t.Fatal("userMessageId missing from prompt response")
	}

	tc.mu.Lock()
	got := tc.updates
	tc.mu.Unlock()
	if len(got) == 0 || got[0] != "Echo: hello world" {
		t.Fatalf("updates = %q", got)
	}

	// An id the client names, as it would after reconnecting to an old thread.
	resumed, err := clientConn.ResumeSession(ctx, acp.ResumeSessionRequest{
		SessionId: "sess_from_before_restart",
		Cwd:       "/tmp",
	})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if len(resumed.ConfigOptions) != 0 {
		t.Fatalf("options = %+v, want none without models configured", resumed.ConfigOptions)
	}
	if _, err := clientConn.Prompt(ctx, acp.PromptRequest{
		SessionId: "sess_from_before_restart",
		Prompt:    []acp.ContentBlock{acp.TextBlock("again")},
	}); err != nil {
		t.Fatalf("prompt after resume: %v", err)
	}

	tc.mu.Lock()
	got = tc.updates
	tc.mu.Unlock()
	if len(got) != 2 || got[1] != "Echo: again" {
		t.Fatalf("updates = %q", got)
	}

	_ = agentConn.Done()
	_ = clientConn.Done()
}
