package agent

import (
	"context"
	"strings"

	"github.com/coder/acp-go-sdk"
)

func (a *Agent) runTurn(ctx context.Context, params acp.PromptRequest) error {
	userText := extractText(params.Prompt)
	if userText == "" {
		userText = "(empty prompt)"
	}

	if err := a.sendUpdate(ctx, params.SessionId, acp.UpdateAgentMessageText("Echo: "+userText)); err != nil {
		return err
	}
	return nil
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
