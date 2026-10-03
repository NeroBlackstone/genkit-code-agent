// Package store persists agent transcripts as JSON, one file per session,
// under a workspace directory keyed by the session's working directory. The
// conversation itself stays in genkit's own message format; the wrapper
// around it carries only agent-side metadata.
package store

import (
	"crypto/sha256"
	"encoding/json/v2"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/firebase/genkit/go/ai"
)

// envelopeVersion is the version this build writes and the only one it reads.
// The messages inside are genkit's format and evolve with genkit, not here.
const envelopeVersion = 1

// SessionMeta is the envelope's agent-side metadata: which model the session
// ran and when it was written. CreatedAt is preserved across saves.
type SessionMeta struct {
	Model     string
	CreatedAt time.Time
	UpdatedAt time.Time
}

// envelope wraps a genkit transcript. Messages stays untouched genkit JSON;
// everything around it is owned by this agent.
type envelope struct {
	Version   int           `json:"version"`
	SessionID string        `json:"sessionId"`
	Model     string        `json:"model,omitempty"`
	CreatedAt time.Time     `json:"createdAt"`
	UpdatedAt time.Time     `json:"updatedAt"`
	Messages  []*ai.Message `json:"messages"`
}

// workspaceMeta is the per-directory index: which cwd this workspace is and
// when it was last touched. Transcripts never depend on it.
type workspaceMeta struct {
	Cwd        string    `json:"cwd"`
	CreatedAt  time.Time `json:"createdAt"`
	LastUsedAt time.Time `json:"lastUsedAt"`
}

// Workspace is one working directory's storage root.
type Workspace struct {
	dir string
	cwd string
}

// Open locates the workspace for cwd under ~/.gca/workspaces. It reads
// nothing and creates nothing: directories appear on the first Save, so
// sessions that never generate leave no trace on disk.
func Open(cwd string) (*Workspace, error) {
	canonical, err := canonicalize(cwd)
	if err != nil {
		return nil, err
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("resolve home dir: %w", err)
	}
	sum := sha256.Sum256([]byte(canonical))
	dir := filepath.Join(home, ".gca", "workspaces", fmt.Sprintf("%x", sum[:8]))
	return &Workspace{dir: dir, cwd: canonical}, nil
}

// canonicalize reduces cwd to an absolute, symlink-free path so the same
// directory always hashes to the same workspace.
func canonicalize(cwd string) (string, error) {
	abs, err := filepath.Abs(cwd)
	if err != nil {
		return "", fmt.Errorf("absolute path: %w", err)
	}
	if resolved, err := filepath.EvalSymlinks(abs); err == nil {
		return resolved, nil
	}
	// A not-yet-existing cwd still gets a stable id from its absolute form.
	return abs, nil
}

// Load reads the stored transcript for id. A missing session is reported as
// fs.ErrNotExist. An unreadable file, a foreign sessionId, or an unknown
// envelope version is quarantined next to the original as .bak and reported
// as an error: the caller starts fresh, the bytes are never destroyed.
func (w *Workspace) Load(id string) ([]*ai.Message, SessionMeta, error) {
	path, err := w.sessionPath(id)
	if err != nil {
		return nil, SessionMeta{}, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, SessionMeta{}, err
	}
	var env envelope
	if err := json.Unmarshal(data, &env); err != nil {
		return nil, SessionMeta{}, quarantine(path, fmt.Errorf("parse session file: %w", err))
	}
	if env.Version != envelopeVersion {
		return nil, SessionMeta{}, quarantine(path, fmt.Errorf("envelope version %d, want %d", env.Version, envelopeVersion))
	}
	if env.SessionID != id {
		return nil, SessionMeta{}, quarantine(path, fmt.Errorf("sessionId %q does not match file for %q", env.SessionID, id))
	}
	meta := SessionMeta{Model: env.Model, CreatedAt: env.CreatedAt, UpdatedAt: env.UpdatedAt}
	return env.Messages, meta, nil
}

// Save writes msgs as id's transcript. The file is written to a temporary
// name and renamed into place, so a reader never sees a half-written file,
// and the returned meta carries the timestamps actually stored. Directories
// are created here, on first use.
func (w *Workspace) Save(id string, meta SessionMeta, msgs []*ai.Message) (SessionMeta, error) {
	path, err := w.sessionPath(id)
	if err != nil {
		return SessionMeta{}, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return SessionMeta{}, fmt.Errorf("create sessions dir: %w", err)
	}
	now := time.Now()
	if meta.CreatedAt.IsZero() {
		meta.CreatedAt = now
	}
	meta.UpdatedAt = now
	data, err := json.Marshal(envelope{
		Version:   envelopeVersion,
		SessionID: id,
		Model:     meta.Model,
		CreatedAt: meta.CreatedAt,
		UpdatedAt: meta.UpdatedAt,
		Messages:  msgs,
	})
	if err != nil {
		return SessionMeta{}, fmt.Errorf("encode session: %w", err)
	}
	// CreateTemp, not a fixed .tmp name, so two writers (two agents on the
	// same cwd) cannot corrupt each other's in-flight file.
	tmp, err := os.CreateTemp(filepath.Dir(path), id+"*.tmp")
	if err != nil {
		return SessionMeta{}, fmt.Errorf("create temp file: %w", err)
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return SessionMeta{}, fmt.Errorf("write session file: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmp.Name())
		return SessionMeta{}, fmt.Errorf("close session file: %w", err)
	}
	if err := os.Chmod(tmp.Name(), 0o600); err != nil {
		os.Remove(tmp.Name())
		return SessionMeta{}, fmt.Errorf("restrict session file: %w", err)
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		os.Remove(tmp.Name())
		return SessionMeta{}, fmt.Errorf("replace session file: %w", err)
	}
	w.touch()
	return meta, nil
}

// sessionPath resolves id to its file. The id reaches the agent from the
// client on resume, so it must not be able to address anything outside the
// sessions directory.
func (w *Workspace) sessionPath(id string) (string, error) {
	if id == "" {
		return "", fmt.Errorf("empty session id")
	}
	for _, r := range id {
		if !(r == '_' || r == '-' ||
			r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9') {
			return "", fmt.Errorf("session id %q contains an unsafe character %q", id, r)
		}
	}
	return filepath.Join(w.dir, "sessions", id+".json"), nil
}

// quarantine renames a file it refuses to read to .bak and returns the cause,
// keeping the bytes for inspection.
func quarantine(path string, cause error) error {
	if err := os.Rename(path, path+".bak"); err != nil {
		return fmt.Errorf("%w (quarantine failed: %v)", cause, err)
	}
	return fmt.Errorf("%w (quarantined to %s.bak)", cause, path)
}

// touch records last use in workspace.json. Best effort: the transcripts are
// what matter, this file is only an index.
func (w *Workspace) touch() {
	path := filepath.Join(w.dir, "workspace.json")
	meta := workspaceMeta{Cwd: w.cwd, CreatedAt: time.Now(), LastUsedAt: time.Now()}
	if data, err := os.ReadFile(path); err == nil {
		var old workspaceMeta
		if json.Unmarshal(data, &old) == nil && !old.CreatedAt.IsZero() {
			meta.CreatedAt = old.CreatedAt
		}
	}
	data, err := json.Marshal(meta)
	if err != nil {
		slog.Warn("encode workspace metadata", "error", err)
		return
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		slog.Warn("write workspace metadata", "path", path, "error", err)
	}
}
