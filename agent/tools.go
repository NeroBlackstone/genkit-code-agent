package agent

import (
	"context"
	"fmt"
	"maps"
	"os"
	"path/filepath"

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

// withCwd carries the session working directory into the tool call through
// genkit's action context, the documented side channel for values the model
// must never see: it rides the ctx given to Generate and reaches tools
// unchanged, so a relative filePath resolves against the session's cwd rather
// than the process's. Existing action context keys are kept, and the map is
// copied first because FromContext hands back the caller's own map.
func withCwd(ctx context.Context, cwd string) context.Context {
	if cwd == "" {
		return ctx
	}
	prev := core.FromContext(ctx)
	merged := make(core.ActionContext, len(prev)+1)
	maps.Copy(merged, prev)
	merged["cwd"] = cwd
	return core.WithActionContext(ctx, merged)
}

// cwdFrom reads the session cwd back out of the action context, falling back
// to the process directory for a call that arrived without one.
func cwdFrom(ctx context.Context) string {
	if cwd, _ := core.FromContext(ctx)["cwd"].(string); cwd != "" {
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
	var write ai.Tool = genkit.LookupTool(g, "write")
	if write == nil {
		write = genkit.DefineTool(g, "write",
			"Writes content to a file, creating parent directories as needed. Overwrites the file if it exists.",
			runWrite,
		)
	}
	return []ai.ToolRef{write}
}

func runWrite(ctx *ai.ToolContext, input WriteInput) (string, error) {
	if input.FilePath == "" {
		return "error: filePath is empty", nil
	}
	path := input.FilePath
	if !filepath.IsAbs(path) {
		path = filepath.Join(cwdFrom(ctx), path)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Sprintf("error: could not create parent directory for %s: %v", path, err), nil
	}
	if err := os.WriteFile(path, []byte(input.Content), 0o644); err != nil {
		return fmt.Sprintf("error: could not write %s: %v", path, err), nil
	}
	return fmt.Sprintf("Wrote %d bytes to %s", len(input.Content), path), nil
}
