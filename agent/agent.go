package agent

import (
	"context"
	"fmt"
	"sync"
	"uuid"

	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

type Session struct {
	Cancel context.CancelFunc
}

type Agent struct {
	conn     *acp.AgentSideConnection
	sessions map[string]*Session
	mu       sync.Mutex

	g     *genkit.Genkit
	model ai.ModelArg
}

var _ acp.Agent = (*Agent)(nil)

func New() *Agent {
	return &Agent{sessions: make(map[string]*Session)}
}

func (a *Agent) SetAgentConnection(conn *acp.AgentSideConnection) {
	a.conn = conn
}

// SetModel wires the genkit instance and model used by runTurn. Without it
// the agent falls back to echoing the prompt.
func (a *Agent) SetModel(g *genkit.Genkit, model ai.ModelArg) {
	a.g = g
	a.model = model
}

func (a *Agent) Initialize(ctx context.Context, params acp.InitializeRequest) (acp.InitializeResponse, error) {
	return acp.InitializeResponse{
		ProtocolVersion: acp.ProtocolVersionNumber,
		AgentInfo: &acp.Implementation{
			Name:    "genkit-code-agent",
			Version: "0.1.0",
		},
		AgentCapabilities: acp.AgentCapabilities{
			LoadSession: false,
			PromptCapabilities: acp.PromptCapabilities{
				Image:           false,
				Audio:           false,
				EmbeddedContext: false,
			},
		},
	}, nil
}

func (a *Agent) NewSession(ctx context.Context, params acp.NewSessionRequest) (acp.NewSessionResponse, error) {
	sid := randomID()
	a.mu.Lock()
	a.sessions[sid] = &Session{}
	a.mu.Unlock()
	return acp.NewSessionResponse{SessionId: acp.SessionId(sid)}, nil
}

func (a *Agent) Prompt(ctx context.Context, params acp.PromptRequest) (acp.PromptResponse, error) {
	sid := string(params.SessionId)
	a.mu.Lock()
	s, ok := a.sessions[sid]
	a.mu.Unlock()
	if !ok {
		return acp.PromptResponse{}, fmt.Errorf("session %s not found", sid)
	}

	if s.Cancel != nil {
		s.Cancel()
	}
	turnCtx, cancel := context.WithCancel(ctx)
	a.mu.Lock()
	s.Cancel = cancel
	a.mu.Unlock()
	defer func() {
		cancel()
		a.mu.Lock()
		s.Cancel = nil
		a.mu.Unlock()
	}()

	if err := a.runTurn(turnCtx, params); err != nil {
		if turnCtx.Err() != nil {
			return acp.PromptResponse{StopReason: acp.StopReasonCancelled}, nil
		}
		return acp.PromptResponse{}, err
	}
	return acp.PromptResponse{StopReason: acp.StopReasonEndTurn}, nil
}

func (a *Agent) Cancel(ctx context.Context, params acp.CancelNotification) error {
	a.mu.Lock()
	s, ok := a.sessions[string(params.SessionId)]
	a.mu.Unlock()
	if ok && s.Cancel != nil {
		s.Cancel()
	}
	return nil
}

func (a *Agent) Authenticate(ctx context.Context, params acp.AuthenticateRequest) (acp.AuthenticateResponse, error) {
	return acp.AuthenticateResponse{}, nil
}

func (a *Agent) Logout(ctx context.Context, params acp.LogoutRequest) (acp.LogoutResponse, error) {
	return acp.LogoutResponse{}, acp.NewMethodNotFound(acp.AgentMethodLogout)
}

func (a *Agent) CloseSession(ctx context.Context, params acp.CloseSessionRequest) (acp.CloseSessionResponse, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	sid := string(params.SessionId)
	if s, ok := a.sessions[sid]; ok {
		if s.Cancel != nil {
			s.Cancel()
		}
		delete(a.sessions, sid)
	}
	return acp.CloseSessionResponse{}, nil
}

func (a *Agent) ListSessions(ctx context.Context, params acp.ListSessionsRequest) (acp.ListSessionsResponse, error) {
	return acp.ListSessionsResponse{}, acp.NewMethodNotFound(acp.AgentMethodSessionList)
}

func (a *Agent) ResumeSession(ctx context.Context, params acp.ResumeSessionRequest) (acp.ResumeSessionResponse, error) {
	return acp.ResumeSessionResponse{}, acp.NewMethodNotFound(acp.AgentMethodSessionResume)
}

func (a *Agent) SetSessionConfigOption(ctx context.Context, params acp.SetSessionConfigOptionRequest) (acp.SetSessionConfigOptionResponse, error) {
	return acp.SetSessionConfigOptionResponse{}, acp.NewMethodNotFound(acp.AgentMethodSessionSetConfigOption)
}

func (a *Agent) SetSessionMode(ctx context.Context, params acp.SetSessionModeRequest) (acp.SetSessionModeResponse, error) {
	return acp.SetSessionModeResponse{}, nil
}

func randomID() string {
	return "sess_" + uuid.New().String()
}
