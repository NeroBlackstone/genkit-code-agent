package agent

import (
	"context"
	"fmt"
	"maps"
	"os"
	"path/filepath"

	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
)

// WriteInput is what the model fills in to call the write tool. The schema
// the model sees is inferred from this type, so the jsonschema_description
// tags are prompt text.
type WriteInput struct {
	FilePath string `json:"filePath" jsonschema_description:"Path of the file to write. Absolute, or relative to the session working directory."`
	Content  string `json:"content" jsonschema_description:"The full content of the file."`
}

// ReadInput is what the model fills in to call the read tool. The schema
// the model sees is inferred from this type, so the jsonschema_description
// tags are prompt text.
type ReadInput struct {
	FilePath string `json:"filePath" jsonschema_description:"Path of the file to read. Absolute, or relative to the session working directory."`
	Line     *int   `json:"line,omitempty" jsonschema_description:"1-based line number to start reading from. Omit to start at the top of the file."`
	Limit    *int   `json:"limit,omitempty" jsonschema_description:"Maximum number of lines to read from that line. Omit to read to the end of the file."`
}

// fileWriter writes one file's whole content over the client's
// fs/write_text_file. writerFor hands one out only when the protocol allows
// the call, and nil when it does not.
type fileWriter func(ctx context.Context, path, content string) error

// fileReader reads one file's contents over the client's
// fs/read_text_file, windowed by the optional line and limit the model
// asked for. readerFor hands one out only when the protocol allows the
// call, and nil when it does not.
type fileReader func(ctx context.Context, path string, line, limit *int) (string, error)

// toolEnv is what one turn gives its tools through genkit's action context,
// the documented side channel for values the model must never see: it rides
// the ctx given to Generate and reaches tools unchanged. Existing action
// context keys are kept, and the map is copied first because FromContext
// hands back the caller's own map.
type toolEnv struct {
	// cwd is the working directory relative paths resolve against; empty
	// means the process directory.
	cwd string
	// read is where tools read: the client's fs/read_text_file, or nil
	// when the client advertised no such capability or no connection is
	// wired, which leaves the tool nowhere to read.
	read fileReader
	// write is where tools write: the client's fs/write_text_file, or nil
	// when the client advertised no such capability or no connection is
	// wired, which leaves the tool nowhere to write.
	write fileWriter
}

// toolEnvKey is the action context key the turn's environment rides under.
const toolEnvKey = "toolEnv"

func withToolEnv(ctx context.Context, env toolEnv) context.Context {
	prev := core.FromContext(ctx)
	merged := make(core.ActionContext, len(prev)+1)
	maps.Copy(merged, prev)
	merged[toolEnvKey] = env
	return core.WithActionContext(ctx, merged)
}

func envFrom(ctx context.Context) toolEnv {
	env, _ := core.FromContext(ctx)[toolEnvKey].(toolEnv)
	return env
}

// cwdFrom reads the session cwd back out of the turn's environment, falling
// back to the process directory for a call that arrived without one.
func cwdFrom(ctx context.Context) string {
	if cwd := envFrom(ctx).cwd; cwd != "" {
		return cwd
	}
	wd, err := os.Getwd()
	if err != nil {
		return ""
	}
	return wd
}

// defineTools registers the agent's tools on g and returns them for
// ai.WithTools. Registration panics on a duplicate name, so an agent sharing
// a genkit instance with an earlier one reuses the already-registered tool.
func defineTools(g *genkit.Genkit) []ai.ToolRef {
	if g == nil {
		return nil
	}
	var read ai.Tool = genkit.LookupTool(g, "read")
	if read == nil {
		read = genkit.DefineTool(g, "read",
			"Reads the contents of a file, including unsaved changes in the editor.",
			runRead,
		)
	}
	var write ai.Tool = genkit.LookupTool(g, "write")
	if write == nil {
		write = genkit.DefineTool(g, "write",
			"Writes content to a file, overwriting it if it exists.",
			runWrite,
		)
	}
	return []ai.ToolRef{read, write}
}

func runWrite(ctx *ai.ToolContext, input WriteInput) (string, error) {
	if input.FilePath == "" {
		return "error: filePath is empty", nil
	}
	// The fs method only takes absolute paths, so a relative one is resolved
	// against the session's working directory here rather than at the client,
	// which was never told which session cwd a tool call meant.
	path := input.FilePath
	if !filepath.IsAbs(path) {
		path = filepath.Join(cwdFrom(ctx), path)
	}
	write := envFrom(ctx).write
	if write == nil {
		// No writer: the client advertised no fs.writeTextFile, and the
		// protocol forbids calling fs/write_text_file anyway, so there is
		// nowhere this tool may put the file.
		return fmt.Sprintf("error: could not write %s: the client does not support fs/write_text_file", path), nil
	}
	if err := write(ctx, path, input.Content); err != nil {
		return fmt.Sprintf("error: could not write %s: %v", path, err), nil
	}
	return fmt.Sprintf("Wrote %d bytes to %s", len(input.Content), path), nil
}

func runRead(ctx *ai.ToolContext, input ReadInput) (string, error) {
	if input.FilePath == "" {
		return "error: filePath is empty", nil
	}
	// As with the write tool: the fs method only takes absolute paths, and
	// the client was never told which session cwd a tool call meant, so a
	// relative one is resolved against the session's working directory here.
	path := input.FilePath
	if !filepath.IsAbs(path) {
		path = filepath.Join(cwdFrom(ctx), path)
	}
	read := envFrom(ctx).read
	if read == nil {
		// No reader: the client advertised no fs.readTextFile, and the
		// protocol forbids calling fs/read_text_file anyway, so there is
		// nowhere this tool may learn the file's contents from.
		return fmt.Sprintf("error: could not read %s: the client does not support fs/read_text_file", path), nil
	}
	content, err := read(ctx, path, input.Line, input.Limit)
	if err != nil {
		return fmt.Sprintf("error: could not read %s: %v", path, err), nil
	}
	return content, nil
}

// writerFor returns the writer this turn's tools write through: the client's
// fs/write_text_file, and nil when the protocol forbids the call, which is
// when the client did not advertise fs.writeTextFile in initialize or no
// connection is wired. Nothing else writes the file in their place: a write
// the client cannot see is the bug ACP's filesystem methods exist to avoid.
func (a *Agent) writerFor(sid string) fileWriter {
	conn := a.conn
	if conn == nil || !a.fsWriteCapable() {
		return nil
	}
	return func(ctx context.Context, path, content string) error {
		_, err := conn.WriteTextFile(ctx, acp.WriteTextFileRequest{
			SessionId: acp.SessionId(sid),
			Path:      path,
			Content:   content,
		})
		return err
	}
}

// readerFor returns the reader this turn's tools read through: the client's
// fs/read_text_file, and nil when the protocol forbids the call, which is
// when the client did not advertise fs.readTextFile in initialize or no
// connection is wired. Nothing reads the file from disk in its place: the
// client's copy, unsaved editor changes included, is the one the user sees.
func (a *Agent) readerFor(sid string) fileReader {
	conn := a.conn
	if conn == nil || !a.fsReadCapable() {
		return nil
	}
	return func(ctx context.Context, path string, line, limit *int) (string, error) {
		resp, err := conn.ReadTextFile(ctx, acp.ReadTextFileRequest{
			SessionId: acp.SessionId(sid),
			Path:      path,
			Line:      line,
			Limit:     limit,
		})
		return resp.Content, err
	}
}
