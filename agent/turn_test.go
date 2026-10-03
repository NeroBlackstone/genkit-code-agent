package agent

import (
	"testing"

	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/ai"
)

func TestUpdateForPart(t *testing.T) {
	const messageID = "msg_test"
	tests := []struct {
		name string
		part *ai.Part
		want acp.SessionUpdate
		ok   bool
	}{
		{
			name: "text becomes an agent message chunk",
			part: ai.NewTextPart("hello"),
			want: acp.UpdateAgentMessageText("hello"),
			ok:   true,
		},
		{
			name: "reasoning becomes a thought chunk",
			part: ai.NewReasoningPart("thinking...", nil),
			want: acp.UpdateAgentThoughtText("thinking..."),
			ok:   true,
		},
		{
			name: "tool request is dropped",
			part: ai.NewToolRequestPart(&ai.ToolRequest{Name: "shell"}),
			ok:   false,
		},
		{
			name: "media is dropped",
			part: ai.NewMediaPart("image/png", "abc"),
			ok:   false,
		},
		{
			name: "empty text is dropped",
			part: &ai.Part{Kind: ai.PartText},
			ok:   false,
		},
		{
			name: "nil part is dropped",
			part: nil,
			ok:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := updateForPart(tt.part, messageID)
			if ok != tt.ok {
				t.Fatalf("ok = %v, want %v", ok, tt.ok)
			}
			if !ok {
				return
			}
			if got.AgentMessageChunk != nil && tt.want.AgentMessageChunk != nil {
				if got.AgentMessageChunk.Content.Text == nil || got.AgentMessageChunk.Content.Text.Text != tt.want.AgentMessageChunk.Content.Text.Text {
					t.Fatalf("update = %+v, want %+v", got, tt.want)
				}
				if got.AgentMessageChunk.MessageId == nil || *got.AgentMessageChunk.MessageId != messageID {
					t.Fatalf("messageId = %v, want %q", got.AgentMessageChunk.MessageId, messageID)
				}
				return
			}
			if got.AgentThoughtChunk != nil && tt.want.AgentThoughtChunk != nil {
				if got.AgentThoughtChunk.Content.Text == nil || got.AgentThoughtChunk.Content.Text.Text != tt.want.AgentThoughtChunk.Content.Text.Text {
					t.Fatalf("update = %+v, want %+v", got, tt.want)
				}
				if got.AgentThoughtChunk.MessageId == nil || *got.AgentThoughtChunk.MessageId != messageID {
					t.Fatalf("messageId = %v, want %q", got.AgentThoughtChunk.MessageId, messageID)
				}
				return
			}
			t.Fatalf("update = %+v, want %+v", got, tt.want)
		})
	}
}
