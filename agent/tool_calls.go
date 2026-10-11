package agent

import (
	"context"
	"os"
	"path/filepath"
	"uuid"

	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/ai"
)

// toolCallReporter is the genkit middleware that reports the model's tool
// requests, and the execution of each one, to the client as ACP tool_call
// updates. WrapModel sees each response the moment the model returns it,
// before genkit runs the tools it asks for, which is the moment ACP calls
// creating: the call is known, its status pending, nothing has executed yet.
// WrapTool then brackets every execution — in_progress as it starts,
// completed with the output the model is about to read back, or failed with
// the error that stopped it. The id minted here rides on the request's Ref,
// so every update for this call arrives on the same toolCallId.
type toolCallReporter struct {
	agent *Agent
	sid   acp.SessionId
	cwd   string
}

func (toolCallReporter) Name() string { return "gca/tool-call-reporter" }

// New builds the per-Generate hook bundle. WrapModel runs once per model
// call in the tool loop, WrapTool once per tool execution (concurrently when
// the model asked for several), so one round of tool requests is reported in
// one place, in order: the calls first, before any of their tools executes,
// each tool's progress and result as it happens.
func (r toolCallReporter) New(context.Context) (*ai.Hooks, error) {
	return &ai.Hooks{WrapModel: r.wrapModel, WrapTool: r.wrapTool}, nil
}

func (r toolCallReporter) wrapModel(ctx context.Context, params *ai.ModelParams, next ai.ModelNext) (*ai.ModelResponse, error) {
	resp, err := next(ctx, params)
	if err != nil {
		return resp, err
	}
	if r.agent == nil || r.agent.conn == nil {
		// Nowhere to report: the turn keeps running, the client simply
		// never learns which tools the model asked for.
		return resp, nil
	}
	for _, part := range resp.ToolRequests() {
		// ACP wants the id unique within the session, which a provider's own
		// call id is not guaranteed to be: local OpenAI-compatible endpoints
		// renumber (call_0, call_1) on every response. A fresh id replaces it
		// before genkit correlates tool responses, and compat_oai only ever
		// echoes Ref back as the call id in the follow-up request, so the
		// rewrite stays self-consistent.
		part.ToolRequest.Ref = "call_" + uuid.New().String()
		if err := r.agent.sendUpdate(ctx, r.sid, toolCallStart(part.ToolRequest, r.cwd)); err != nil {
			return resp, err
		}
	}
	return resp, nil
}

// wrapTool reports one execution under the id its creating update was filed
// under: in_progress before the tool runs, then completed with the output
// the model is about to see, or failed with the error that stopped it.
// genkit runs one goroutine per tool of a round, so parallel calls interleave
// on the wire; each call's own updates stay in order, and the connection
// serializes the writes.
func (r toolCallReporter) wrapTool(ctx context.Context, params *ai.ToolParams, next ai.ToolNext) (*ai.MultipartToolResponse, error) {
	// A request without an id names no call on the client to update, and
	// neither does a reporter with nowhere to report to.
	if r.agent == nil || r.agent.conn == nil || params.Request == nil || params.Request.Ref == "" {
		return next(ctx, params)
	}
	ref := params.Request.Ref
	if err := r.agent.sendUpdate(ctx, r.sid, toolCallOutcome(ref, acp.ToolCallStatusInProgress, nil)); err != nil {
		return nil, err
	}
	resp, err := next(ctx, params)
	if err != nil {
		if held, _ := ai.IsToolInterruptError(err); held {
			// Held, not failed: the call comes back through a resume that
			// either runs it (and this hook reports the run then) or
			// answers it (and the approval loop closes it). Marking it
			// failed here would close a call that is about to reopen under
			// the same id, so the client keeps the pending it has.
			return resp, err
		}
		// The tool's own error is what the turn fails on; the report of it
		// rides along, dropped only when it cannot be sent either.
		_ = r.agent.sendUpdate(ctx, r.sid, toolCallOutcome(ref, acp.ToolCallStatusFailed, err.Error()))
		return resp, err
	}
	var output any
	if resp != nil {
		output = resp.Output
	}
	if err := r.agent.sendUpdate(ctx, r.sid, toolCallOutcome(ref, acp.ToolCallStatusCompleted, output)); err != nil {
		// The call ran but the client never learned it did; the turn stops
		// rather than stream an answer under a call still pending.
		return nil, err
	}
	return resp, nil
}

// toolCallStart builds the creating update for one tool request: the pending
// status no execution has started under yet, the raw input the model sent,
// and the file the call acts on when it names one. The request's Ref is the
// id the update is filed under: the live path mints one before reporting it
// (see wrapModel), a replay reads the one the call was stored with, so a
// call keeps one toolCallId from its first appearance onward. Ref is never
// empty on either path — genkit fills in a missing one before a response
// leaves the model (ensureToolRequestRefs).
func toolCallStart(req *ai.ToolRequest, cwd string) acp.SessionUpdate {
	view := toolCallViewFor(req.Name)
	path := resolveToolPath(inputString(req.Input, view.pathKey), cwd)

	var locations []acp.ToolCallLocation
	if path != "" {
		location := acp.ToolCallLocation{Path: path}
		if line, ok := inputInt(req.Input, "line"); ok {
			location.Line = line
		}
		locations = append(locations, location)
	}
	// Built by hand rather than through the SDK's start helpers: those
	// mirror a single location into rawInput.path when it is missing, which
	// would write into the model's own input map on its way to the tool.
	return acp.SessionUpdate{ToolCall: &acp.SessionUpdateToolCall{
		ToolCallId: acp.ToolCallId(req.Ref),
		Title:      view.title(req.Name, path),
		Kind:       view.kind,
		Status:     acp.ToolCallStatusPending,
		RawInput:   req.Input,
		Locations:  locations,
	}}
}

// toolCallOutcome builds the tool_call_update carrying one call's result:
// the status it reached and, once it has one, the output it produced. The id
// is the request's Ref on every path — the one wrapModel minted live, the one
// the call was stored with on replay — so the update lands on the call the
// client already has. One builder for both paths is what keeps a replayed
// result identical to the one the live turn reported.
func toolCallOutcome(ref string, status acp.ToolCallStatus, output any) acp.SessionUpdate {
	return acp.SessionUpdate{ToolCallUpdate: &acp.SessionToolCallUpdate{
		ToolCallId: acp.ToolCallId(ref),
		Status:     &status,
		RawOutput:  output,
	}}
}

// toolCallView is how one of the agent's tools reads to the client: the
// kind its icon follows, the verb its title opens with, and the input field
// naming the file the call acts on, when there is one.
type toolCallView struct {
	kind    acp.ToolKind
	verb    string
	pathKey string
}

func toolCallViewFor(name string) toolCallView {
	switch name {
	case "read":
		return toolCallView{kind: acp.ToolKindRead, verb: "Reading", pathKey: "filePath"}
	case "write":
		return toolCallView{kind: acp.ToolKindEdit, verb: "Writing", pathKey: "filePath"}
	}
	return toolCallView{kind: acp.ToolKindOther}
}

// title reads "Reading /abs/path" when the tool names a file, the bare
// verb when it does not, and is the tool's own name for a tool this agent
// knows nothing about.
func (v toolCallView) title(name, path string) string {
	switch {
	case v.verb == "":
		return name
	case path != "":
		return v.verb + " " + path
	default:
		return v.verb + " file"
	}
}

// resolveToolPath turns the path the model named into the absolute one the
// client's follow-along expects: a relative one lands under the session cwd,
// the same rule the read and write tools resolve by.
func resolveToolPath(path, cwd string) string {
	if path == "" || filepath.IsAbs(path) {
		return path
	}
	if cwd == "" {
		wd, err := os.Getwd()
		if err != nil {
			return path
		}
		cwd = wd
	}
	return filepath.Join(cwd, path)
}

// inputString reads one string field out of a tool request's decoded input.
// Arguments arrive JSON-decoded, so anything of the wrong shape reads as
// absent rather than as an error: this feeds display metadata, and a field
// it cannot parse must not stop the turn.
func inputString(input any, key string) string {
	m, _ := input.(map[string]any)
	s, _ := m[key].(string)
	return s
}

// inputInt reads one integer field out of a tool request's decoded input,
// under the same read-as-absent rule as inputString. JSON numbers decode as
// float64, which is the shape the model's arguments arrive in.
func inputInt(input any, key string) (*int, bool) {
	m, _ := input.(map[string]any)
	switch v := m[key].(type) {
	case float64:
		n := int(v)
		return &n, true
	case int:
		return &v, true
	}
	return nil, false
}
