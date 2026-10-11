package agent

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"uuid"

	"genkit-code-agent/store"

	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

const systemPrompt = `You are a coding agent. You help users with software engineering tasks:
reading and explaining code, writing and editing files, debugging and running commands,
answering technical questions. Use the read tool to inspect files and the write tool
to create or overwrite files. Be direct and concise. When you change code,
follow the existing project conventions. Never invent file paths or APIs: verify
them before referring to them.`

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

	turnCtx, modelRef, release := s.claimTurn(ctx)
	defer release()

	if err := a.runTurn(turnCtx, params, s, modelRef, userMsgID); err != nil {
		// A cancelled turn reports stopReason cancelled rather than
		// failing: the protocol ties cancellation to a prompt the client
		// itself stopped, and a permission outcome of the same name ends
		// the turn the same way — with nothing run and nothing stored.
		if turnCtx.Err() != nil || errors.Is(err, errPermissionCancelled) {
			return acp.PromptResponse{StopReason: acp.StopReasonCancelled, UserMessageId: &userMsgID}, nil
		}
		return acp.PromptResponse{}, err
	}
	return acp.PromptResponse{StopReason: acp.StopReasonEndTurn, UserMessageId: &userMsgID}, nil
}

func (a *Agent) Cancel(ctx context.Context, params acp.CancelNotification) error {
	if s, ok := a.get(string(params.SessionId)); ok {
		s.cancelTurn()
	}
	return nil
}

// runTurn answers one prompt on the model the session selected, falling back
// to echoing when no model was configured. The conversation is read from the
// store at the start of the turn and written back at the end, so a session
// keeps nothing of it in between; the turn lock covers both, so the next
// prompt on this session reads what this one wrote. A failed or cancelled
// turn writes nothing, leaving no dangling question or half-written answer
// for the next one to replay to the model.
func (a *Agent) runTurn(ctx context.Context, params acp.PromptRequest, s *Session, modelRef, userMsgID string) error {
	userText := extractText(params.Prompt)
	if userText == "" {
		userText = "(empty prompt)"
	}

	model, ok := a.modelFor(modelRef)
	if a.g == nil || !ok {
		return a.sendUpdate(ctx, params.SessionId, acp.UpdateAgentMessageText("Echo: "+userText))
	}

	s.turn.Lock()
	defer s.turn.Unlock()

	// A prompt that arrived behind an earlier one cancels it, then waits for
	// the lock here; by the time it gets in, its own turn may already have
	// been superseded too, and there is no point reading a history it will
	// never write.
	if err := ctx.Err(); err != nil {
		return err
	}

	history, meta, err := s.load(string(params.SessionId))
	if err != nil {
		return err
	}

	// One messageId groups every update of this turn, thinking and answer
	// alike, which is how Kilo's ACP output labels them and how a client
	// knows the chunks belong to one message.
	messageID := "msg_" + uuid.New().String()

	// The turn's tools get their environment here: the session cwd relative
	// paths resolve against, and the reader and writer that take files to
	// and from this client over fs/read_text_file and fs/write_text_file,
	// or none when the client cannot take them. The same context carries
	// the resume below, so the tools re-run under approval have the same
	// environment as the ones the first call ran.
	envCtx := withToolEnv(ctx, toolEnv{
		cwd:   s.cwd,
		read:  a.readerFor(string(params.SessionId)),
		write: a.writerFor(string(params.SessionId)),
	})

	stream := func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		for _, part := range chunk.Content {
			update, ok := updateForPart(part, messageID)
			if !ok {
				continue
			}
			if err := a.sendUpdate(ctx, params.SessionId, update); err != nil {
				return err
			}
		}
		return nil
	}

	resp, err := genkit.Generate(envCtx, a.g,
		ai.WithModel(ai.NewModelRef(model.Ref, model.Config)),
		ai.WithSystem(systemPrompt),
		ai.WithMessages(history...),
		ai.WithPrompt(userText),
		ai.WithTools(a.tools...),
		// The bundle leads with the approval gate: the client hears about
		// each tool call through the reporter's WrapModel hook the moment
		// the model requests it, pending and before execution, but a call
		// the gate holds never reaches the reporter's WrapTool — it stays
		// pending until the resume either runs it or answers it.
		ai.WithUse(a.turnMiddleware(s, params.SessionId)...),
		ai.WithStreaming(stream),
	)
	if err != nil {
		return err
	}

	// A held tool ends the call with FinishReason interrupted and its
	// requests unexecuted. The approval loop puts each one to the client
	// over session/request_permission and runs Generate again over the
	// same conversation with the decisions attached, until the model
	// stops pausing; what follows only ever sees a finished turn.
	resp, err = a.resolveInterrupts(envCtx, resp, params.SessionId, s,
		func(cur *ai.ModelResponse, restarts, responds []*ai.Part) (*ai.ModelResponse, error) {
			msgs := cur.History()
			// The system prompt re-travels as WithSystem rather than as
			// the history message it already is: doubling it would put
			// two system messages in front of the provider.
			if len(msgs) > 0 && msgs[0].Role == ai.RoleSystem {
				msgs = msgs[1:]
			}
			return genkit.Generate(envCtx, a.g,
				ai.WithModel(ai.NewModelRef(model.Ref, model.Config)),
				ai.WithSystem(systemPrompt),
				ai.WithMessages(msgs...),
				ai.WithTools(a.tools...),
				// The same middleware the first call carried, rebuilt so
				// an allow_always answered earlier this turn is in the
				// gate before this call's own tool rounds run: resume
				// keeps the same config as the call it resumes, plus
				// whatever the session has remembered since.
				ai.WithUse(a.turnMiddleware(s, params.SessionId)...),
				ai.WithStreaming(stream),
				ai.WithToolRestarts(restarts...),
				ai.WithToolResponses(responds...),
			)
		})
	if err != nil {
		return err
	}
	if resp == nil || resp.Message == nil || resp.Message.Text() == "" {
		// Nothing usable came back: skip the turn rather than storing a
		// question the model never answered.
		return nil
	}

	// Both ids ride in genkit's own metadata field: the user message keeps
	// the id the client named (or one assigned for it), the answer keeps the
	// msg_ id its chunks streamed under, so a replay can regroup both.
	// The transcript genkit hands back is the whole conversation as the last
	// turn saw it: the system prompt, the stored history, this prompt, and
	// every tool round in between. The system prompt is re-sent each turn
	// and belongs to no session, so it is dropped; what remains is this
	// turn's exchange, appended to the history it ran on.
	stored := resp.History()
	if len(stored) > 0 && stored[0].Role == ai.RoleSystem {
		stored = stored[1:]
	}
	stored = append([]*ai.Message{}, stored...)
	if len(stored) > len(history) {
		stored[len(history)].Metadata = map[string]any{"messageId": userMsgID}
	}
	if resp.Message.Metadata == nil {
		resp.Message.Metadata = map[string]any{}
	}
	resp.Message.Metadata["messageId"] = messageID

	if s.ws == nil {
		// Nowhere to store it: the turn happened, the next one simply runs
		// without it.
		return nil
	}
	// The whole conversation is written back, this turn's exchange included.
	// A failure is logged rather than raised: the answer already streamed to
	// the client, so the turn stands, and the store error surfaces as a
	// shorter conversation on the next one.
	if _, err := s.ws.Save(string(params.SessionId), store.SessionMeta{
		Model:     modelRef,
		CreatedAt: meta.CreatedAt,
	}, stored); err != nil {
		slog.Warn("persist transcript", "session", params.SessionId, "error", err)
	}
	return nil
}

// updatesForReplay turns one stored message back into the ACP updates its
// original turn streamed: user text as user_message_chunk, an answer's
// reasoning first and then its text, a model message's tool requests as the
// tool_call updates the reporter sent when it saw them, and a tool message's
// responses as the tool_call_update that closed each call, all carrying the
// ids they were saved under so the client regroups the replay exactly as the
// live turn was grouped. The live turn's in_progress update has no stored
// counterpart — the transcript holds results, not progress — so a replay runs
// a call straight from pending to its stored outcome. A message stored before
// ids existed gets a fresh one for the replay.
func updatesForReplay(m *ai.Message, cwd string) []acp.SessionUpdate {
	if m == nil {
		return nil
	}
	savedID, _ := m.Metadata["messageId"].(string)
	switch m.Role {
	case ai.RoleUser:
		var out []acp.SessionUpdate
		for _, part := range m.Content {
			if part == nil || part.Kind != ai.PartText || part.Text == "" {
				continue
			}
			update := acp.UpdateUserMessageText(part.Text)
			if savedID != "" {
				update.UserMessageChunk.MessageId = &savedID
			}
			out = append(out, update)
		}
		return out
	case ai.RoleModel:
		id := savedID
		if id == "" {
			id = "msg_" + uuid.New().String()
		}
		var out []acp.SessionUpdate
		for _, part := range m.Content {
			if part != nil && part.IsToolRequest() && part.ToolRequest != nil {
				// The call comes back as its own creating update rather
				// than as a chunk of this message: it is grouped under the
				// id it was stored with, not under the message id the text
				// around it rides on.
				out = append(out, toolCallStart(part.ToolRequest, cwd))
				continue
			}
			if update, ok := updateForPart(part, id); ok {
				out = append(out, update)
			}
		}
		return out
	case ai.RoleTool:
		// The results answering the calls above: status and output,
		// rebuilt from the stored response through the builder the
		// middleware reports live with, on the id the creating update was
		// filed under, so a loaded session closes every call it made
		// rather than leaving them pending.
		var out []acp.SessionUpdate
		for _, part := range m.Content {
			if part == nil || part.ToolResponse == nil || part.ToolResponse.Ref == "" {
				continue
			}
			// A response the agent answered on the user's refusal carries
			// the interruptResponse marker Respond stamps; it replays as
			// the failed the live turn reported, not as a run that
			// completed.
			status := acp.ToolCallStatusCompleted
			if part.Metadata["interruptResponse"] != nil {
				status = acp.ToolCallStatusFailed
			}
			out = append(out, toolCallOutcome(part.ToolResponse.Ref, status, part.ToolResponse.Output))
		}
		return out
	}
	return nil
}

// updateForPart maps a Genkit part to the ACP update that carries it, stamped
// with messageID so every chunk of the turn groups under one message. Text is
// the agent's reply; reasoning is the model's thinking, which ACP streams as a
// separate thought channel. Tool requests are dropped rather than mapped:
// the streaming path would report a call the reporter already reported from
// the assembled response, with half its arguments, and the replay path turns
// them into tool_call updates of its own before calling this. Other kinds
// (media, data) have no ACP update wired up yet and are dropped.
func updateForPart(part *ai.Part, messageID string) (acp.SessionUpdate, bool) {
	if part == nil || part.Text == "" {
		return acp.SessionUpdate{}, false
	}
	switch part.Kind {
	case ai.PartText:
		update := acp.UpdateAgentMessageText(part.Text)
		update.AgentMessageChunk.MessageId = &messageID
		return update, true
	case ai.PartReasoning:
		update := acp.UpdateAgentThoughtText(part.Text)
		update.AgentThoughtChunk.MessageId = &messageID
		return update, true
	default:
		return acp.SessionUpdate{}, false
	}
}

func (a *Agent) sendUpdate(ctx context.Context, sid acp.SessionId, update acp.SessionUpdate) error {
	if a.conn == nil {
		return errors.New("agent connection not set")
	}
	return a.conn.SessionUpdate(ctx, acp.SessionNotification{
		SessionId: sid,
		Update:    update,
	})
}

func extractText(blocks []acp.ContentBlock) string {
	var sb strings.Builder
	for _, b := range blocks {
		if b.Text != nil {
			sb.WriteString(b.Text.Text)
		}
	}
	return sb.String()
}
