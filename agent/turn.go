package agent

import (
	"context"
	"errors"
	"log/slog"
	"strings"
	"uuid"

	"genkit-code-agent/store"

	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

const systemPrompt = `You are a coding agent. You help users with software engineering tasks:
reading and explaining code, writing and editing files, debugging, running commands,
and answering technical questions. Use the write tool to create or overwrite files;
it creates parent directories for you. Be direct and concise. When you change code,
follow the existing project conventions. Never invent file paths or APIs: verify
them before referring to them.`

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

	resp, err := genkit.Generate(withCwd(ctx, s.cwd), a.g,
		ai.WithModel(ai.NewModelRef(model.Ref, model.Config)),
		ai.WithSystem(systemPrompt),
		ai.WithMessages(history...),
		ai.WithPrompt(userText),
		ai.WithTools(a.tools...),
		ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
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
		}),
	)
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
// reasoning first and then its text, each carrying the id it was saved under
// so the client regroups the replay exactly as the live turn was grouped.
// A message stored before ids existed gets a fresh one for the replay.
func updatesForReplay(m *ai.Message) []acp.SessionUpdate {
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
			if update, ok := updateForPart(part, id); ok {
				out = append(out, update)
			}
		}
		return out
	}
	return nil
}

// updateForPart maps a Genkit part to the ACP update that carries it, stamped
// with messageID so every chunk of the turn groups under one message. Text is
// the agent's reply; reasoning is the model's thinking, which ACP streams as a
// separate thought channel. Other kinds (tool calls, media, data) have no ACP
// update wired up yet and are dropped.
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
