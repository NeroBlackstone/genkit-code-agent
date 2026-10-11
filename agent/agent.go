package agent

import (
	"context"
	"sync"

	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/genkit"
)

type Agent struct {
	conn     *acp.AgentSideConnection
	sessions map[string]*Session
	// mu guards the sessions map, and only the map: no store or model work
	// is done under it, so an unrelated session never waits on this lock.
	mu sync.RWMutex

	g      *genkit.Genkit
	models []Model
}

var (
	_ acp.Agent       = (*Agent)(nil)
	_ acp.AgentLoader = (*Agent)(nil)
)

// New wires the genkit instance and the models runTurn picks from: they become
// the model config option, and the first one is a new session's value. A nil g
// with no models leaves the agent falling back to echoing the prompt.
func New(g *genkit.Genkit, models []Model) *Agent {
	return &Agent{
		sessions: make(map[string]*Session),
		g:        g,
		models:   models,
	}
}

// SetAgentConnection injects the connection updates are sent over. It cannot
// ride New because the connection is built around the agent itself; every path
// that sends an update must run after it.
func (a *Agent) SetAgentConnection(conn *acp.AgentSideConnection) {
	a.conn = conn
}

func (a *Agent) Initialize(ctx context.Context, params acp.InitializeRequest) (acp.InitializeResponse, error) {
	// The client's capabilities ride in on params and are deliberately not
	// stored: the filesystem half of them gated fs/read_text_file and
	// fs/write_text_file, methods ACP v2 removes. File and command access
	// now belong to the agent — file tools through genkit's Filesystem
	// middleware, rooted at the session cwd, and an MCP server when a
	// client wants to offer more — so nothing here needs to remember what
	// the client said it supports.
	return acp.InitializeResponse{
		ProtocolVersion: acp.ProtocolVersionNumber,
		AgentInfo: &acp.Implementation{
			Name:    "genkit-code-agent",
			Version: "0.1.0",
		},
		AgentCapabilities: acp.AgentCapabilities{
			// LoadSession is served from transcripts stored on disk: session/load
			// restores the session and streams its history back as notifications.
			// Resume stays the lighter way in, for a client that already has the
			// history itself and is what Zed falls back to when loadSession is off.
			LoadSession: true,
			SessionCapabilities: acp.SessionCapabilities{
				Resume: &acp.SessionResumeCapabilities{},
			},
			PromptCapabilities: acp.PromptCapabilities{
				Image:           false,
				Audio:           false,
				EmbeddedContext: false,
			},
		},
	}, nil
}

func (a *Agent) Authenticate(ctx context.Context, params acp.AuthenticateRequest) (acp.AuthenticateResponse, error) {
	return acp.AuthenticateResponse{}, nil
}

func (a *Agent) Logout(ctx context.Context, params acp.LogoutRequest) (acp.LogoutResponse, error) {
	return acp.LogoutResponse{}, acp.NewMethodNotFound(acp.AgentMethodLogout)
}

func (a *Agent) ListSessions(ctx context.Context, params acp.ListSessionsRequest) (acp.ListSessionsResponse, error) {
	return acp.ListSessionsResponse{}, acp.NewMethodNotFound(acp.AgentMethodSessionList)
}

func (a *Agent) SetSessionMode(ctx context.Context, params acp.SetSessionModeRequest) (acp.SetSessionModeResponse, error) {
	return acp.SetSessionModeResponse{}, nil
}
