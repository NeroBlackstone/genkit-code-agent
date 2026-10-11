package oai

import (
	"encoding/json/v2"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/openai/openai-go/option"
)

// capturedToolRound is the shape DeepSeek rejected: an assistant message with
// tool_calls and no reasoning_content, in a request that ends with the tool's
// answer.
const capturedToolRound = `{
  "model": "deepseek-flash",
  "stream": true,
  "messages": [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": [{"type": "text", "text": "read notes.txt"}]},
    {"role": "assistant", "content": "I'll read the file first.",
     "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "read", "arguments": "{\"filePath\":\"notes.txt\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "\"file body\""}
  ],
  "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
}`

// readBody runs padAssistantReasoning over body and fails the test when it
// cannot parse.
func readBody(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("unmarshal padded body: %v", err)
	}
	return out
}

// messagesOf reads the messages array out of a parsed request body.
func messagesOf(t *testing.T, body map[string]any) []map[string]any {
	t.Helper()
	raw, ok := body["messages"].([]any)
	if !ok {
		t.Fatalf("messages = %#v, want an array", body["messages"])
	}
	msgs := make([]map[string]any, 0, len(raw))
	for _, m := range raw {
		msg, ok := m.(map[string]any)
		if !ok {
			t.Fatalf("message = %#v, want an object", m)
		}
		msgs = append(msgs, msg)
	}
	return msgs
}

// TestPadAssistantReasoningPadsToolRound covers the failing request: the
// assistant message gains reasoning_content "" and nothing else moves.
func TestPadAssistantReasoningPadsToolRound(t *testing.T) {
	padded, err := padAssistantReasoning([]byte(capturedToolRound))
	if err != nil {
		t.Fatalf("pad: %v", err)
	}
	body := readBody(t, padded)
	if body["model"] != "deepseek-flash" || body["stream"] != true {
		t.Errorf("envelope fields lost: model=%v stream=%v", body["model"], body["stream"])
	}
	if tools, ok := body["tools"].([]any); !ok || len(tools) != 1 {
		t.Errorf("tools = %#v, want the original single entry", body["tools"])
	}
	msgs := messagesOf(t, body)
	roles := make([]string, len(msgs))
	for i, m := range msgs {
		roles[i], _ = m["role"].(string)
	}
	if got := strings.Join(roles, ","); got != "system,user,assistant,tool" {
		t.Fatalf("roles = %q, want the original order", got)
	}
	assistant := msgs[2]
	if r, ok := assistant["reasoning_content"].(string); !ok || r != "" {
		t.Errorf("assistant reasoning_content = %#v, want empty string", assistant["reasoning_content"])
	}
	if _, ok := assistant["tool_calls"]; !ok {
		t.Errorf("assistant tool_calls lost: %#v", assistant)
	}
}

// TestPadAssistantReasoningLeavesPaddedMessagesAlone pins the two cases the
// padding must not touch: a message that already carries the field, and
// non-assistant roles.
func TestPadAssistantReasoningLeavesPaddedMessagesAlone(t *testing.T) {
	in := `{
  "messages": [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "thinking", "reasoning_content": "the model's CoT"},
    {"role": "tool", "tool_call_id": "c", "content": "ok"}
  ]
}`
	padded, err := padAssistantReasoning([]byte(in))
	if err != nil {
		t.Fatalf("pad: %v", err)
	}
	msgs := messagesOf(t, readBody(t, padded))
	if got := msgs[1]["reasoning_content"]; got != "the model's CoT" {
		t.Errorf("reasoning_content = %#v, want it preserved", got)
	}
	if _, ok := msgs[0]["reasoning_content"]; ok {
		t.Errorf("user message gained reasoning_content: %#v", msgs[0])
	}
	if _, ok := msgs[2]["reasoning_content"]; ok {
		t.Errorf("tool message gained reasoning_content: %#v", msgs[2])
	}
}

// TestPadAssistantReasoningPassthrough checks bodies the padder declines:
// they must come back byte-identical so the request goes out as written.
func TestPadAssistantReasoningPassthrough(t *testing.T) {
	cases := map[string]string{
		"not JSON":      "this is not json",
		"no messages":   `{"model":"deepseek-flash"}`,
		"messages type": `{"messages":"nope"}`,
	}
	for name, in := range cases {
		t.Run(name, func(t *testing.T) {
			out, err := padAssistantReasoning([]byte(in))
			if err == nil && string(out) != in {
				t.Errorf("body changed:\n got %q\nwant %q", out, in)
			}
		})
	}
}

// recordingNext is the middleware's downstream: it captures the request as it
// leaves and answers with a stub completion.
func recordingNext(t *testing.T, got *[]byte) option.MiddlewareNext {
	t.Helper()
	return func(req *http.Request) (*http.Response, error) {
		body, err := io.ReadAll(req.Body)
		if err != nil {
			t.Fatalf("read forwarded body: %v", err)
		}
		*got = body
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"choices":[]}`)),
			Request:    req,
		}, nil
	}
}

// TestPadAssistantReasoningMiddleware is the end-to-end middleware shape:
// the body reaching the transport carries the padding.
func TestPadAssistantReasoningMiddleware(t *testing.T) {
	var got []byte
	req, err := http.NewRequest(http.MethodPost, "https://api.deepseek.com/chat/completions",
		strings.NewReader(capturedToolRound))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PadAssistantReasoning(req, recordingNext(t, &got)); err != nil {
		t.Fatalf("middleware: %v", err)
	}
	if req.ContentLength != int64(len(got)) {
		t.Errorf("ContentLength = %d, want the padded body's %d", req.ContentLength, len(got))
	}
	assistant := messagesOf(t, readBody(t, got))[2]
	if r, ok := assistant["reasoning_content"].(string); !ok || r != "" {
		t.Errorf("assistant reasoning_content = %#v, want empty string at the transport", assistant["reasoning_content"])
	}
}

// TestPadAssistantReasoningSkipsOtherEndpoints keeps the padder off requests
// it was not written for: a GET or a non-chat-completions path goes out with
// its body untouched.
func TestPadAssistantReasoningSkipsOtherEndpoints(t *testing.T) {
	for _, tt := range []struct {
		name   string
		method string
		url    string
	}{
		{"GET", http.MethodGet, "https://api.deepseek.com/chat/completions"},
		{"models list", http.MethodPost, "https://api.deepseek.com/models"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var got []byte
			original := capturedToolRound
			req, err := http.NewRequest(tt.method, tt.url, strings.NewReader(original))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := PadAssistantReasoning(req, recordingNext(t, &got)); err != nil {
				t.Fatalf("middleware: %v", err)
			}
			if string(got) != original {
				t.Errorf("body rewritten on %s %s", tt.method, tt.url)
			}
		})
	}
}
