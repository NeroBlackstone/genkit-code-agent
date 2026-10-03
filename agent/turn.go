package agent

import (
	"context"
	"strings"

	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

func (a *Agent) runTurn(ctx context.Context, params acp.PromptRequest) error {
	userText := extractText(params.Prompt)
	if userText == "" {
		userText = "(empty prompt)"
	}

	if a.g == nil || a.model == nil {
		return a.sendUpdate(ctx, params.SessionId, acp.UpdateAgentMessageText("Echo: "+userText))
	}

	_, err := genkit.Generate(ctx, a.g,
		ai.WithModel(a.model),
		ai.WithPrompt(userText),
		ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
			for _, part := range chunk.Content {
				if part.Kind != ai.PartText || part.Text == "" {
					continue
				}
				if err := a.sendUpdate(ctx, params.SessionId, acp.UpdateAgentMessageText(part.Text)); err != nil {
					return err
				}
			}
			return nil
		}),
	)
	return err
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
