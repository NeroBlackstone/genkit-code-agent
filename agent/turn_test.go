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

// TestUpdatesForReplayRebuildsToolCall replays one stored tool round the way
// session/load hands it back: the prompt and the answer as the chunks they
// streamed under their saved ids, the model's tool request as the creating
// tool_call the reporter sent, filed under the ref it was stored with, and
// the tool's response as the completed tool_call_update that closed it —
// same id, the status execution finished under, the output the model read
// back. The live turn's in_progress has no stored counterpart, so the replay
// runs the call from pending straight to completed.
func TestUpdatesForReplayRebuildsToolCall(t *testing.T) {
	const cwd = "/workspace"
	stored := []*ai.Message{
		ai.NewUserTextMessage("show me main.go"),
		{Role: ai.RoleModel, Content: []*ai.Part{ai.NewToolRequestPart(&ai.ToolRequest{
			Ref:   "call_stored",
			Name:  "read",
			Input: map[string]any{"filePath": "src/main.go", "line": float64(7)},
		})}},
		{Role: ai.RoleTool, Content: []*ai.Part{ai.NewToolResponsePart(&ai.ToolResponse{
			Name: "read", Ref: "call_stored", Output: "file body",
		})}},
		{Role: ai.RoleModel, Metadata: map[string]any{"messageId": "msg_answer"},
			Content: []*ai.Part{ai.NewTextPart("here it is")}},
	}

	byRole := make([][]acp.SessionUpdate, len(stored))
	for i, m := range stored {
		byRole[i] = updatesForReplay(m, cwd)
	}

	if got := byRole[0]; len(got) != 1 || got[0].UserMessageChunk == nil ||
		got[0].UserMessageChunk.Content.Text.Text != "show me main.go" {
		t.Fatalf("prompt replay = %+v, want one user_message_chunk", got)
	}

	replayed := byRole[1]
	if len(replayed) != 1 {
		t.Fatalf("tool request replay = %+v, want exactly one update", replayed)
	}
	toolCall := replayed[0].ToolCall
	if toolCall == nil {
		t.Fatalf("update = %+v, want a tool_call", replayed[0])
	}
	if toolCall.ToolCallId != "call_stored" {
		t.Errorf("toolCallId = %q, want the id it was stored with", toolCall.ToolCallId)
	}
	if toolCall.Status != acp.ToolCallStatusPending {
		t.Errorf("status = %q, want pending", toolCall.Status)
	}
	if toolCall.Kind != acp.ToolKindRead {
		t.Errorf("kind = %q, want read", toolCall.Kind)
	}
	if want := "Reading /workspace/src/main.go"; toolCall.Title != want {
		t.Errorf("title = %q, want %q", toolCall.Title, want)
	}
	if len(toolCall.Locations) != 1 || toolCall.Locations[0].Path != "/workspace/src/main.go" {
		t.Errorf("locations = %+v, want the file the call names", toolCall.Locations)
	}
	if raw, ok := toolCall.RawInput.(map[string]any); !ok || raw["filePath"] != "src/main.go" {
		t.Errorf("rawInput = %#v, want the model's own arguments", toolCall.RawInput)
	}

	// The response closes the call the message before it opened: the same
	// id, the status execution finished under, and the output the model
	// read back from the tool.
	results := byRole[2]
	if len(results) != 1 {
		t.Fatalf("tool response replay = %+v, want exactly one update", results)
	}
	outcome := results[0].ToolCallUpdate
	if outcome == nil {
		t.Fatalf("update = %+v, want a tool_call_update", results[0])
	}
	if outcome.ToolCallId != "call_stored" {
		t.Errorf("toolCallId = %q, want the id the call was created under", outcome.ToolCallId)
	}
	if outcome.Status == nil || *outcome.Status != acp.ToolCallStatusCompleted {
		t.Errorf("status = %v, want completed", outcome.Status)
	}
	if outcome.RawOutput != "file body" {
		t.Errorf("rawOutput = %#v, want the stored output", outcome.RawOutput)
	}

	answer := byRole[3]
	if len(answer) != 1 || answer[0].AgentMessageChunk == nil ||
		answer[0].AgentMessageChunk.Content.Text.Text != "here it is" {
		t.Fatalf("answer replay = %+v, want one agent_message_chunk", answer)
	}
	if id := answer[0].AgentMessageChunk.MessageId; id == nil || *id != "msg_answer" {
		t.Errorf("answer messageId = %v, want the saved msg_answer", id)
	}
}
