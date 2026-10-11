package agent

import (
	"context"
	"sync"

	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/ai"
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
	// tools are the tools this agent offers the model, registered on g when
	// one is wired and empty otherwise.
	tools []ai.ToolRef

	// capsMu guards caps: Initialize writes it on the connection's dispatch
	// goroutine, a turn reads it on another, so the read side takes the lock
	// instead of leaning on initialize happening first.
	capsMu sync.RWMutex
	// caps is what the client advertised in initialize. Only its filesystem
	// half is consulted today, by writerFor and readerFor, which must know
	// it before calling fs/write_text_file or fs/read_text_file.
	caps acp.ClientCapabilities
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
		tools:    defineTools(g),
	}
}

// SetAgentConnection injects the connection updates are sent over. It cannot
// ride New because the connection is built around the agent itself; every path
// that sends an update must run after it.
func (a *Agent) SetAgentConnection(conn *acp.AgentSideConnection) {
	a.conn = conn
}

func (a *Agent) Initialize(ctx context.Context, params acp.InitializeRequest) (acp.InitializeResponse, error) {
	// The client's half of the capability negotiation, kept so the methods
	// it gates can be checked: fs/write_text_file may only be called once
	// the client has said it supports it.
	a.capsMu.Lock()
	a.caps = params.ClientCapabilities
	a.capsMu.Unlock()

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

// fsWriteCapable reports whether the client advertised fs.writeTextFile.
// writerFor consults it before every turn decides where its writes go.
func (a *Agent) fsWriteCapable() bool {
	a.capsMu.RLock()
	defer a.capsMu.RUnlock()
	return a.caps.Fs.WriteTextFile
}

// fsReadCapable reports whether the client advertised fs.readTextFile.
// readerFor consults it before every turn decides where its reads go.
func (a *Agent) fsReadCapable() bool {
	a.capsMu.RLock()
	defer a.capsMu.RUnlock()
	return a.caps.Fs.ReadTextFile
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
