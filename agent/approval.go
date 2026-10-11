package agent

import (
	"context"
	"errors"
	"fmt"

	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/middleware"
)

// The four outcomes a session/request_permission exchange may select,
// offered as the option ids the client echoes back. The always kinds are
// remembered on the session (see Session.approved and Session.denied); the
// once kinds leave no trace beyond the round they resolve.
const (
	permissionAllowOnce    acp.PermissionOptionId = "allow-once"
	permissionAllowAlways  acp.PermissionOptionId = "allow-always"
	permissionRejectOnce   acp.PermissionOptionId = "reject-once"
	permissionRejectAlways acp.PermissionOptionId = "reject-always"
)

// denialReply is what the model reads back for a tool call the user refused:
// a response rather than silence, since a tool request nothing answers is a
// conversation no provider accepts.
const denialReply = "The user denied this tool call."

// maxAutoDeniedRounds bounds a run in which no prompt goes out at all: every
// interrupt resolved from a rejection the session already remembered. A
// model that keeps re-asking for a tool the user chose always to reject
// would otherwise spin between interrupt and auto-denial forever, since
// each resume resets genkit's own turn counter. One prompt resets the
// count: a user clicking through rounds is the flow this bounds against
// nothing.
const maxAutoDeniedRounds = 3

// errPermissionCancelled is what the client's "cancelled" outcome turns
// into: the prompt turn it answers is over, with nothing run and nothing
// stored, which is how Prompt reports it back.
var errPermissionCancelled = errors.New("permission request cancelled")

// turnMiddleware builds the middleware bundle for one Generate call: the
// approval gate first, the tool-call reporter second. The chain applies
// outer-to-inner, so the gate decides before the reporter's WrapTool ever
// sees the call — a held tool stays pending on the client instead of
// flicking through in_progress, and the reporter's Ref is already on the
// request the permission prompt names. The bundle is rebuilt per call
// rather than held for the turn: an allow_always answered mid-turn must be
// in the gate before the next Generate runs.
func (a *Agent) turnMiddleware(s *Session, sid acp.SessionId) []ai.Middleware {
	return []ai.Middleware{
		&middleware.ToolApproval{AllowedTools: s.allowedTools()},
		toolCallReporter{agent: a, sid: sid, cwd: s.cwd},
	}
}

// resumeFunc re-runs Generate over a conversation the approval gate paused:
// the whole history as it stands, the restarts for the calls the user
// approved, and the responses for the ones refused. It is a func so the
// turn can build it around its own model, streaming, and middleware while
// resolveInterrupts stays about the decisions.
type resumeFunc func(cur *ai.ModelResponse, restarts, responds []*ai.Part) (*ai.ModelResponse, error)

// resolveInterrupts runs the approval loop: for every interrupt resp came
// back with, ask for a decision, attach it as a restart (approved) or a
// response (refused), and re-run Generate through resume. It returns when
// the conversation stops pausing — the model answered, or something broke.
func (a *Agent) resolveInterrupts(ctx context.Context, resp *ai.ModelResponse, sid acp.SessionId, s *Session, resume resumeFunc) (*ai.ModelResponse, error) {
	autoDeniedRounds := 0
	for resp != nil && resp.FinishReason == ai.FinishReasonInterrupted {
		var (
			restarts []*ai.Part
			responds []*ai.Part
			prompted bool
		)
		for _, interrupt := range resp.Interrupts() {
			req := interrupt.ToolRequest
			if req == nil {
				continue
			}
			allow, asked, err := a.decidePermission(ctx, sid, s, req)
			if err != nil {
				return nil, err
			}
			prompted = prompted || asked

			tool := genkit.LookupTool(a.g, req.Name)
			if allow {
				if tool == nil {
					return nil, fmt.Errorf("approve unknown tool %q", req.Name)
				}
				// The approval flag rides the restart metadata: it is the
				// only thing the gate reads as approval, so a bare resume
				// of the same conversation cannot slip past it.
				restarts = append(restarts, tool.Restart(interrupt, &ai.RestartOptions{
					ResumedMetadata: map[string]any{"toolApproved": true},
				}))
				continue
			}
			// Refused: the model hears why instead of waiting on a
			// response that never comes, and the client's call is closed
			// rather than left pending forever.
			if tool != nil {
				responds = append(responds, tool.Respond(interrupt, denialReply, nil))
			} else {
				responds = append(responds, ai.NewToolResponsePart(&ai.ToolResponse{
					Name:   req.Name,
					Ref:    req.Ref,
					Output: denialReply,
				}))
			}
			if err := a.sendUpdate(ctx, sid, toolCallOutcome(req.Ref, acp.ToolCallStatusFailed, denialReply)); err != nil {
				return nil, err
			}
		}
		if len(restarts) == 0 && len(responds) == 0 {
			return nil, errors.New("interrupted response carried no tool request to resolve")
		}
		if prompted {
			autoDeniedRounds = 0
		} else if autoDeniedRounds++; autoDeniedRounds > maxAutoDeniedRounds {
			return nil, fmt.Errorf("tool calls hit a remembered rejection %d rounds in a row", autoDeniedRounds-1)
		}

		next, err := resume(resp, restarts, responds)
		if err != nil {
			return nil, err
		}
		resp = next
	}
	return resp, nil
}

// decidePermission resolves one interrupted tool request: a rejection the
// session already remembers answers it without prompting, otherwise the
// client is asked over session/request_permission. prompted reports whether
// a request actually went out, which is what the loop's anti-spin counter
// counts rounds by. A cancelled outcome is not a refusal: it ends the turn
// (errPermissionCancelled), since the protocol ties it to a cancelled
// prompt turn that must run nothing further.
func (a *Agent) decidePermission(ctx context.Context, sid acp.SessionId, s *Session, req *ai.ToolRequest) (allow, prompted bool, err error) {
	if s.toolDenied(req.Name) {
		return false, false, nil
	}
	if a.conn == nil {
		return false, false, fmt.Errorf("no client connection to approve tool %q", req.Name)
	}
	resp, err := a.conn.RequestPermission(ctx, acp.RequestPermissionRequest{
		SessionId: sid,
		// Only the id: the client already has the call's details from the
		// tool_call update the reporter sent when the model requested it.
		ToolCall: acp.ToolCallUpdate{ToolCallId: acp.ToolCallId(req.Ref)},
		Options: []acp.PermissionOption{
			{OptionId: permissionAllowOnce, Name: "Allow once", Kind: acp.PermissionOptionKindAllowOnce},
			{OptionId: permissionAllowAlways, Name: "Always allow", Kind: acp.PermissionOptionKindAllowAlways},
			{OptionId: permissionRejectOnce, Name: "Reject", Kind: acp.PermissionOptionKindRejectOnce},
			{OptionId: permissionRejectAlways, Name: "Always reject", Kind: acp.PermissionOptionKindRejectAlways},
		},
	})
	if err != nil {
		return false, false, fmt.Errorf("ask permission for tool %q: %w", req.Name, err)
	}
	if resp.Outcome.Cancelled != nil {
		if ctx.Err() != nil {
			return false, false, ctx.Err()
		}
		return false, false, errPermissionCancelled
	}
	if resp.Outcome.Selected == nil {
		return false, false, fmt.Errorf("permission response for tool %q carried no outcome", req.Name)
	}
	switch resp.Outcome.Selected.OptionId {
	case permissionAllowAlways:
		s.rememberApproval(req.Name)
		return true, true, nil
	case permissionAllowOnce:
		return true, true, nil
	case permissionRejectAlways:
		s.rememberDenial(req.Name)
		return false, true, nil
	default:
		// reject-once, and any option id a client made up: nothing is
		// approved unless it says so.
		return false, true, nil
	}
}
