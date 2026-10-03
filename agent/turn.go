package agent

import (
	"context"
	"strings"

	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

const systemPrompt = `You are a coding agent. You help users with software engineering tasks:
reading and explaining code, writing and editing files, debugging, running commands,
and answering technical questions. Be direct and concise. When you change code,
follow the existing project conventions. Never invent file paths or APIs: verify
them before referring to them.`

// runTurn answers one prompt on the model the session selected, falling back
// to echoing when no model was configured.
func (a *Agent) runTurn(ctx context.Context, params acp.PromptRequest, modelRef string) error {
	userText := extractText(params.Prompt)
	if userText == "" {
		userText = "(empty prompt)"
	}

	model, ok := a.modelFor(modelRef)
	if a.g == nil || !ok {
		return a.sendUpdate(ctx, params.SessionId, acp.UpdateAgentMessageText("Echo: "+userText))
	}

	_, err := genkit.Generate(ctx, a.g,
		ai.WithModel(ai.NewModelRef(model.Ref, model.Config)),
		ai.WithSystem(systemPrompt),
		ai.WithPrompt(userText),
		ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
			for _, part := range chunk.Content {
				update, ok := updateForPart(part)
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
	return err
}

// updateForPart maps a Genkit part to the ACP update that carries it. Text is
// the agent's reply; reasoning is the model's thinking, which ACP streams as a
// separate thought channel. Other kinds (tool calls, media, data) have no ACP
// update wired up yet and are dropped.
func updateForPart(part *ai.Part) (acp.SessionUpdate, bool) {
	if part == nil || part.Text == "" {
		return acp.SessionUpdate{}, false
	}
	switch part.Kind {
	case ai.PartText:
		return acp.UpdateAgentMessageText(part.Text), true
	case ai.PartReasoning:
		return acp.UpdateAgentThoughtText(part.Text), true
	default:
		return acp.SessionUpdate{}, false
	}
}

func (a *Agent) sendUpdate(ctx context.Context, sid acp.SessionId, update acp.SessionUpdate) error {
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
