package agent

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"sync"
	"uuid"

	"genkit-code-agent/store"

	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// modelConfigOptionID is the id of the model selector this agent offers.
const modelConfigOptionID acp.SessionConfigId = "model"

// Model is a model the agent can run. Ref names it for genkit and is the value
// the model config option carries, so a selection needs no mapping table. Group
// names the provider and must be set: it is the selector's group header.
type Model struct {
	Ref    string
	Name   string
	Group  string
	Config any
}

// Session is one conversation's live state. Its content is not here: the
// transcript lives in the store, which is the single source of truth, and is
// read and written around each turn. What stays in memory is what a running
// process must hold: where to find the conversation, which model it runs, and
// the handle to the turn in flight.
type Session struct {
	// turn serializes this session's load -> generate -> save pipeline. A
	// prompt holds it for the whole turn, so a later one cannot read a
	// transcript the earlier has not written back yet. Cancel never takes
	// it: interrupting a turn must not queue behind it.
	turn sync.Mutex

	// mu guards the fields below. It is never held across a store read, a
	// model call, or a wait on turn.
	mu sync.Mutex
	// cancel stops the turn currently running on this session, if any.
	cancel context.CancelFunc
	// turnSeq numbers the turns that claim cancel, so a turn superseded by a
	// newer one clears nothing but its own handle on the way out.
	turnSeq int
	// model is the ref of the model this session runs, taken from the model
	// config option and defaulting to the first configured model.
	model string

	// ws is this session's workspace, nil when the cwd gave no workspace to
	// store a conversation in; a session without one runs turn by turn with
	// no history, there being nowhere to keep it. ready is closed once ws and
	// the stored model are settled, and every path that hands out this
	// session waits on it first, which is what lets ws go without a lock.
	ws    *store.Workspace
	ready chan struct{}

	// cwd is the working directory the session was opened with. Tools
	// resolve relative paths against it; like ws it is settled before ready
	// closes, so it needs no lock.
	cwd string
}

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

func (a *Agent) NewSession(ctx context.Context, params acp.NewSessionRequest) (acp.NewSessionResponse, error) {
	sid := randomID()
	s := a.openSession(sid, params.Cwd)
	return acp.NewSessionResponse{
		SessionId:     acp.SessionId(sid),
		ConfigOptions: a.configOptions(s.currentModel()),
	}, nil
}

// ResumeSession continues a session by the id the client names, without
// replaying history, which is what the resume capability promises. The model
// the session last ran on comes back with it, an id from before this process
// started included; an unknown id is registered here with the default model
// and picks up its stored conversation on the next turn.
func (a *Agent) ResumeSession(ctx context.Context, params acp.ResumeSessionRequest) (acp.ResumeSessionResponse, error) {
	s := a.openSession(string(params.SessionId), params.Cwd)
	return acp.ResumeSessionResponse{ConfigOptions: a.configOptions(s.currentModel())}, nil
}

// LoadSession restores a stored session and streams its whole transcript back
// to the client as notifications, which is what session/load asks the agent
// to do. The transcript is read from the store here rather than from memory,
// so a session loaded without ever being prompted replays just the same.
func (a *Agent) LoadSession(ctx context.Context, params acp.LoadSessionRequest) (acp.LoadSessionResponse, error) {
	s := a.openSession(string(params.SessionId), params.Cwd)
	msgs, _, err := s.load(string(params.SessionId))
	if err != nil {
		return acp.LoadSessionResponse{}, err
	}
	for _, m := range msgs {
		for _, update := range updatesForReplay(m) {
			if err := a.sendUpdate(ctx, params.SessionId, update); err != nil {
				return acp.LoadSessionResponse{}, err
			}
		}
	}
	return acp.LoadSessionResponse{ConfigOptions: a.configOptions(s.currentModel())}, nil
}

// openSession returns the session for sid, registering it with the default
// model first when it is not registered yet, and pointing it at the workspace
// stored for it in cwd. The workspace is attached off Agent.mu: opening one
// reads the disk, and a prompt on another session must not wait on it.
func (a *Agent) openSession(sid, cwd string) *Session {
	a.mu.Lock()
	if s, ok := a.sessions[sid]; ok {
		a.mu.Unlock()
		<-s.ready
		return s
	}
	s := &Session{ready: make(chan struct{}), cwd: cwd}
	if len(a.models) > 0 {
		s.model = a.models[0].Ref
	}
	a.sessions[sid] = s
	a.mu.Unlock()

	// Closing ready releases every caller waiting on this session, so it
	// must happen whatever attachStore concludes.
	defer close(s.ready)
	if cwd != "" {
		a.attachStore(s, sid, cwd)
	}
	return s
}

// get returns the session registered under sid, once its workspace is
// attached, or reports that no session is registered for it.
func (a *Agent) get(sid string) (*Session, bool) {
	a.mu.RLock()
	s, ok := a.sessions[sid]
	a.mu.RUnlock()
	if !ok {
		return nil, false
	}
	<-s.ready
	return s, true
}

// attachStore opens cwd's workspace and takes from it the model the session
// last ran on. The transcript itself is deliberately left on disk: it is read
// by the turn and the replay that need it, never cached here. A workspace
// that cannot be opened leaves the session running without history, and a
// stored session that cannot be read keeps the default model, the file having
// been quarantined by the store.
func (a *Agent) attachStore(s *Session, sid, cwd string) {
	ws, err := store.Open(cwd)
	if err != nil {
		slog.Warn("workspace unavailable, sessions will run without history", "cwd", cwd, "error", err)
		return
	}
	s.ws = ws
	_, meta, err := ws.Load(sid)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		// A new session: nothing stored yet, which is the normal case.
		return
	case err != nil:
		slog.Warn("stored session unusable, keeping the default model", "session", sid, "error", err)
		return
	}
	if meta.Model != "" {
		if _, ok := a.modelFor(meta.Model); ok {
			s.model = meta.Model
		}
	}
}

// load reads this session's transcript from the store, the only place a
// conversation exists. A session with no workspace has no history, and a
// session nothing was ever saved for is the same case rather than an error.
// Any other failure is returned instead of falling back to an empty
// conversation: starting empty here would mean saving over a file that may
// still hold the real one.
func (s *Session) load(sid string) ([]*ai.Message, store.SessionMeta, error) {
	if s.ws == nil {
		return nil, store.SessionMeta{}, nil
	}
	msgs, meta, err := s.ws.Load(sid)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return nil, store.SessionMeta{}, nil
	case err != nil:
		return nil, store.SessionMeta{}, fmt.Errorf("read stored transcript for %s: %w", sid, err)
	}
	return msgs, meta, nil
}

// currentModel returns the model this session runs.
func (s *Session) currentModel() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.model
}

func (a *Agent) Prompt(ctx context.Context, params acp.PromptRequest) (acp.PromptResponse, error) {
	sid := string(params.SessionId)
	s, ok := a.get(sid)
	if !ok {
		return acp.PromptResponse{}, fmt.Errorf("session %s not found", sid)
	}

	// The id this turn's user message goes under: the client's own when it
	// named one, ours otherwise, so the response can always echo an id and
	// the turn can stamp it into the stored transcript either way.
	userMsgID := uuid.New().String()
	if params.MessageId != nil && *params.MessageId != "" {
		userMsgID = *params.MessageId
	}

	// Claim the session before waiting for the turn lock: the turn already
	// running is cancelled first, so it unwinds instead of holding the
	// pipeline this one is about to enter. The sequence number marks the
	// handle as ours, so the superseded turn cannot clear it on its way out.
	s.mu.Lock()
	if s.cancel != nil {
		s.cancel()
	}
	s.turnSeq++
	seq := s.turnSeq
	turnCtx, cancel := context.WithCancel(ctx)
	s.cancel = cancel
	// Taken now so a selection made while this turn runs applies to the next
	// one rather than switching models mid-generation.
	modelRef := s.model
	s.mu.Unlock()

	defer func() {
		cancel()
		s.mu.Lock()
		if s.turnSeq == seq {
			s.cancel = nil
		}
		s.mu.Unlock()
	}()

	if err := a.runTurn(turnCtx, params, s, modelRef, userMsgID); err != nil {
		if turnCtx.Err() != nil {
			return acp.PromptResponse{StopReason: acp.StopReasonCancelled, UserMessageId: &userMsgID}, nil
		}
		return acp.PromptResponse{}, err
	}
	return acp.PromptResponse{StopReason: acp.StopReasonEndTurn, UserMessageId: &userMsgID}, nil
}

func (a *Agent) Cancel(ctx context.Context, params acp.CancelNotification) error {
	if s, ok := a.get(string(params.SessionId)); ok {
		s.mu.Lock()
		if s.cancel != nil {
			s.cancel()
		}
		s.mu.Unlock()
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
	sid := string(params.SessionId)
	a.mu.Lock()
	s, ok := a.sessions[sid]
	if ok {
		delete(a.sessions, sid)
	}
	a.mu.Unlock()
	if !ok {
		return acp.CloseSessionResponse{}, nil
	}
	// Stopping the turn in flight is all close does to it: the conversation
	// is on disk and stays there, waiting to be resumed.
	<-s.ready
	s.mu.Lock()
	if s.cancel != nil {
		s.cancel()
	}
	s.mu.Unlock()
	return acp.CloseSessionResponse{}, nil
}

func (a *Agent) ListSessions(ctx context.Context, params acp.ListSessionsRequest) (acp.ListSessionsResponse, error) {
	return acp.ListSessionsResponse{}, acp.NewMethodNotFound(acp.AgentMethodSessionList)
}

// SetSessionConfigOption applies a client's change to a session config option.
// ACP answers with the complete option list rather than the change, so the
// response is rebuilt from the session's new state.
func (a *Agent) SetSessionConfigOption(ctx context.Context, params acp.SetSessionConfigOptionRequest) (acp.SetSessionConfigOptionResponse, error) {
	// Only select options are offered; a boolean one would need the client to
	// have advertised support for it first, so it cannot arrive here.
	if params.ValueId == nil {
		return acp.SetSessionConfigOptionResponse{}, fmt.Errorf("unsupported config option type")
	}
	if params.ValueId.ConfigId != modelConfigOptionID {
		return acp.SetSessionConfigOptionResponse{}, fmt.Errorf("unknown config option %q", params.ValueId.ConfigId)
	}
	value := string(params.ValueId.Value)
	if _, ok := a.modelFor(value); !ok {
		return acp.SetSessionConfigOptionResponse{}, fmt.Errorf("unknown model %q", value)
	}
	s, ok := a.get(string(params.ValueId.SessionId))
	if !ok {
		return acp.SetSessionConfigOptionResponse{}, fmt.Errorf("session %s not found", params.ValueId.SessionId)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.model = value
	return acp.SetSessionConfigOptionResponse{ConfigOptions: a.configOptions(s.model)}, nil
}

func (a *Agent) SetSessionMode(ctx context.Context, params acp.SetSessionModeRequest) (acp.SetSessionModeResponse, error) {
	return acp.SetSessionModeResponse{}, nil
}

// configOptions builds the complete config state ACP carries: the whole option
// list with its current values, which session/new and session/set_config_option
// both return rather than a bare change.
func (a *Agent) configOptions(current string) []acp.SessionConfigOption {
	if len(a.models) == 0 {
		return []acp.SessionConfigOption{}
	}
	category := acp.SessionConfigOptionCategoryModel
	return []acp.SessionConfigOption{{
		Select: &acp.SessionConfigOptionSelect{
			Id:           modelConfigOptionID,
			Name:         "Model",
			Category:     &category,
			Type:         "select",
			CurrentValue: acp.SessionConfigValueId(current),
			Options:      a.modelOptions(),
		},
	}}
}

// modelOptions lists the configured models as select values grouped by their
// provider, one group per distinct Group in configuration order.
func (a *Agent) modelOptions() acp.SessionConfigSelectOptions {
	var groups []acp.SessionConfigSelectGroup
	at := map[string]int{}
	for _, m := range a.models {
		value := acp.SessionConfigSelectOption{
			Value: acp.SessionConfigValueId(m.Ref),
			Name:  m.Name,
		}
		if i, ok := at[m.Group]; ok {
			groups[i].Options = append(groups[i].Options, value)
			continue
		}
		at[m.Group] = len(groups)
		groups = append(groups, acp.SessionConfigSelectGroup{
			Group:   acp.SessionConfigGroupId(m.Group),
			Name:    m.Group,
			Options: []acp.SessionConfigSelectOption{value},
		})
	}
	boxed := acp.SessionConfigSelectOptionsGrouped(groups)
	return acp.SessionConfigSelectOptions{Grouped: &boxed}
}

// modelFor returns the configured model with that ref.
func (a *Agent) modelFor(ref string) (Model, bool) {
	for _, m := range a.models {
		if m.Ref == ref {
			return m, true
		}
	}
	return Model{}, false
}

func randomID() string {
	return "sess_" + uuid.New().String()
}
