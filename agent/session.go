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
)

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

	// approved and denied hold the always-choices the user made on this
	// session's permission prompts: allow_always and reject_always. They
	// live only here — nothing writes them to the store or to the config —
	// so a new session asks again and a restart forgets. mu guards them
	// like model: a permission prompt resolves while the turn lock is
	// held, but the maps outlive the turn that wrote them.
	approved map[string]bool
	denied   map[string]bool

	// ws is this session's workspace, nil when the cwd gave no workspace to
	// store a conversation in; a session without one runs turn by turn with
	// no history, there being nowhere to keep it. ready is closed once ws and
	// the stored model are settled, and every path that hands out this
	// session waits on it first, which is what lets ws go without a lock.
	ws    *store.Workspace
	ready chan struct{}

	// cwd is the working directory the session was opened with. The
	// filesystem middleware roots the model's file tools at it and the
	// tool-call display resolves relative paths against it; like ws it is
	// settled before ready closes, so it needs no lock.
	cwd string
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
		for _, update := range updatesForReplay(m, s.cwd) {
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
	s.cancelTurn()
	return acp.CloseSessionResponse{}, nil
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

// allowedTools lists the tools an always-allow has cleared for the rest of
// the session: the names the approval gate lets through without asking.
// Nil when nothing has been cleared, which the gate treats the same as an
// empty list — everything interrupts.
func (s *Session) allowedTools() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.approved) == 0 {
		return nil
	}
	tools := make([]string, 0, len(s.approved))
	for name := range s.approved {
		tools = append(tools, name)
	}
	return tools
}

// toolDenied reports whether the user chose always-reject for this tool on
// this session, which answers the next request for it without prompting.
func (s *Session) toolDenied(name string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.denied[name]
}

// rememberApproval records an always-allow choice for the rest of the
// session: from the next call on, the gate never interrupts this tool.
func (s *Session) rememberApproval(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.approved == nil {
		s.approved = make(map[string]bool)
	}
	s.approved[name] = true
}

// rememberDenial records an always-reject choice for the rest of the
// session: the next request for this tool is answered with the denial
// rather than put to the user again.
func (s *Session) rememberDenial(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.denied == nil {
		s.denied = make(map[string]bool)
	}
	s.denied[name] = true
}

// claimTurn hands the session to a new turn before the turn lock is waited
// for: the turn already running is cancelled first, so it unwinds instead of
// holding the pipeline this one is about to enter. The sequence number marks
// the handle as ours, so a superseded turn cannot clear it on its way out.
// modelRef is read in the same critical section, so a selection made while
// this turn runs applies to the next one rather than switching models
// mid-generation. The returned release clears only this claim, and must be
// called on the way out.
func (s *Session) claimTurn(ctx context.Context) (turnCtx context.Context, modelRef string, release func()) {
	s.mu.Lock()
	if s.cancel != nil {
		s.cancel()
	}
	s.turnSeq++
	seq := s.turnSeq
	turnCtx, cancel := context.WithCancel(ctx)
	s.cancel = cancel
	modelRef = s.model
	s.mu.Unlock()

	release = func() {
		cancel()
		s.mu.Lock()
		if s.turnSeq == seq {
			s.cancel = nil
		}
		s.mu.Unlock()
	}
	return turnCtx, modelRef, release
}

// cancelTurn stops the turn currently running on this session, if any.
func (s *Session) cancelTurn() {
	s.mu.Lock()
	if s.cancel != nil {
		s.cancel()
	}
	s.mu.Unlock()
}

func randomID() string {
	return "sess_" + uuid.New().String()
}
