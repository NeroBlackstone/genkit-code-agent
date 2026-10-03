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

type Session struct {
	Cancel context.CancelFunc
	// model is the ref of the model this session runs, taken from the model
	// config option and defaulting to the first configured model.
	model string
	// ws persists this session's transcript, nil when the cwd gave no
	// workspace to write to.
	ws *store.Workspace
	// meta is the stored envelope's metadata, refreshed on every save.
	meta store.SessionMeta
	// transcript is the full conversation, oldest first, guarded by
	// Agent.mu and written to disk after every successful turn. Only its
	// most recent window is ever replayed to the model; session/load can
	// replay all of it.
	transcript []*ai.Message
}

type Agent struct {
	conn     *acp.AgentSideConnection
	sessions map[string]*Session
	mu       sync.Mutex

	g      *genkit.Genkit
	models []Model
}

var (
	_ acp.Agent       = (*Agent)(nil)
	_ acp.AgentLoader = (*Agent)(nil)
)

func New() *Agent {
	return &Agent{sessions: make(map[string]*Session)}
}

func (a *Agent) SetAgentConnection(conn *acp.AgentSideConnection) {
	a.conn = conn
}

// SetModels wires the genkit instance and the models runTurn picks from: they
// become the model config option, and the first one is a new session's value.
// Without it the agent falls back to echoing the prompt.
func (a *Agent) SetModels(g *genkit.Genkit, models []Model) {
	a.g = g
	a.models = models
}

func (a *Agent) Initialize(ctx context.Context, params acp.InitializeRequest) (acp.InitializeResponse, error) {
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

func (a *Agent) NewSession(ctx context.Context, params acp.NewSessionRequest) (acp.NewSessionResponse, error) {
	sid := randomID()
	s := a.openSession(sid, params.Cwd)
	return acp.NewSessionResponse{
		SessionId:     acp.SessionId(sid),
		ConfigOptions: a.configOptions(s.model),
	}, nil
}

// ResumeSession continues a session by the id the client names, without
// replaying history, which is what the resume capability promises. A stored
// transcript for that id is still loaded, so an id from before this process
// started keeps its conversation and the model it ran on; an unknown id is
// registered here with the default model.
func (a *Agent) ResumeSession(ctx context.Context, params acp.ResumeSessionRequest) (acp.ResumeSessionResponse, error) {
	s := a.openSession(string(params.SessionId), params.Cwd)
	return acp.ResumeSessionResponse{ConfigOptions: a.configOptions(s.model)}, nil
}

// LoadSession restores a stored session and streams its whole transcript back
// to the client as notifications, which is what session/load asks the agent
// to do now that transcripts live on disk.
func (a *Agent) LoadSession(ctx context.Context, params acp.LoadSessionRequest) (acp.LoadSessionResponse, error) {
	s := a.openSession(string(params.SessionId), params.Cwd)
	a.mu.Lock()
	msgs := append([]*ai.Message(nil), s.transcript...)
	model := s.model
	a.mu.Unlock()
	for _, m := range msgs {
		for _, update := range updatesForReplay(m) {
			if err := a.sendUpdate(ctx, params.SessionId, update); err != nil {
				return acp.LoadSessionResponse{}, err
			}
		}
	}
	return acp.LoadSessionResponse{ConfigOptions: a.configOptions(model)}, nil
}

// openSession returns the session for sid, registering it with the default
// model first when it is not registered yet, and adopting the transcript
// stored for it in cwd's workspace when one exists.
func (a *Agent) openSession(sid, cwd string) *Session {
	a.mu.Lock()
	defer a.mu.Unlock()
	if s, ok := a.sessions[sid]; ok {
		return s
	}
	s := &Session{}
	if len(a.models) > 0 {
		s.model = a.models[0].Ref
	}
	if cwd != "" {
		a.attachStore(s, sid, cwd)
	}
	a.sessions[sid] = s
	return s
}

// attachStore opens cwd's workspace and, when a stored transcript for sid
// exists, adopts it along with the model that session ran on. Callers hold
// a.mu. A workspace that cannot be opened leaves the session in memory only;
// a transcript that cannot be read is reported and skipped, the file having
// been quarantined by the store.
func (a *Agent) attachStore(s *Session, sid, cwd string) {
	ws, err := store.Open(cwd)
	if err != nil {
		slog.Warn("workspace unavailable, session will not persist", "cwd", cwd, "error", err)
		return
	}
	s.ws = ws
	msgs, meta, err := ws.Load(sid)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		// A new session: nothing stored yet, which is the normal case.
		return
	case err != nil:
		// Quarantined or unreadable: keep the session, start the transcript
		// empty, and say so; the store preserved the original bytes.
		slog.Warn("stored transcript unusable, starting empty", "session", sid, "error", err)
		return
	}
	s.transcript, s.meta = msgs, meta
	if meta.Model != "" {
		if _, ok := a.modelFor(meta.Model); ok {
			s.model = meta.Model
		}
	}
}

func (a *Agent) Prompt(ctx context.Context, params acp.PromptRequest) (acp.PromptResponse, error) {
	sid := string(params.SessionId)
	a.mu.Lock()
	s, ok := a.sessions[sid]
	var modelRef string
	if ok {
		// Taken now so a selection made while this turn runs applies to the
		// next one rather than switching models mid-generation.
		modelRef = s.model
	}
	a.mu.Unlock()
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

	if err := a.runTurn(turnCtx, params, s, modelRef, userMsgID); err != nil {
		if turnCtx.Err() != nil {
			return acp.PromptResponse{StopReason: acp.StopReasonCancelled, UserMessageId: &userMsgID}, nil
		}
		return acp.PromptResponse{}, err
	}
	return acp.PromptResponse{StopReason: acp.StopReasonEndTurn, UserMessageId: &userMsgID}, nil
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

// SetSessionConfigOption applies a client's change to a session config option.
// ACP answers with the complete option list rather than the change, so the
// response is rebuilt from the session's new state.
func (a *Agent) SetSessionConfigOption(ctx context.Context, params acp.SetSessionConfigOptionRequest) (acp.SetSessionConfigOptionResponse, error) {
	// Only select options are offered; a boolean one would need the client to
	// have advertised support for it first, so it cannot arrive here.
	if params.ValueId == nil {
		return acp.SetSessionConfigOptionResponse{}, fmt.Errorf("unsupported config option type")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	sid := string(params.ValueId.SessionId)
	s, ok := a.sessions[sid]
	if !ok {
		return acp.SetSessionConfigOptionResponse{}, fmt.Errorf("session %s not found", sid)
	}
	if params.ValueId.ConfigId != modelConfigOptionID {
		return acp.SetSessionConfigOptionResponse{}, fmt.Errorf("unknown config option %q", params.ValueId.ConfigId)
	}
	value := string(params.ValueId.Value)
	if _, ok := a.modelFor(value); !ok {
		return acp.SetSessionConfigOptionResponse{}, fmt.Errorf("unknown model %q", value)
	}
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
