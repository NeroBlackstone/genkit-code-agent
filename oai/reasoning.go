// Package oai patches OpenAI-compatible chat requests on their way to the
// provider, for rules genkit's compat_oai serialization layer does not follow.
package oai

import (
	"bytes"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"io"
	"net/http"
	"strings"

	"github.com/openai/openai-go/option"
)

// PadAssistantReasoning is an openai-go request middleware that adds
// `"reasoning_content": ""` to every assistant message lacking the field.
//
// DeepSeek's thinking mode requires the reasoning_content of all previous
// turns to be passed back whenever the request carries tools: a missing field
// is answered with 400 "The `reasoning_content` in the thinking mode must be
// passed back to the API", while an empty string is accepted. compat_oai only
// emits the field when the message carries a non-empty reasoning part, so a
// turn the model answered without a chain of thought — or a transcript saved
// before reasoning was kept — serializes without it and the next tool round
// fails. Padding at the HTTP boundary fixes both without altering the
// conversation genkit replays.
func PadAssistantReasoning(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	if req.Method != http.MethodPost || !strings.HasSuffix(req.URL.Path, "/chat/completions") || req.Body == nil {
		return next(req)
	}
	body, err := io.ReadAll(req.Body)
	req.Body.Close()
	if err != nil {
		return nil, err
	}
	// A body that is not the JSON shape handled here is sent untouched:
	// padding is an optimization against one provider rule, never a reason
	// to fail a request that would otherwise go out as written.
	padded, err := padAssistantReasoning(body)
	if err != nil {
		padded = body
	}
	req.Body = io.NopCloser(bytes.NewReader(padded))
	req.ContentLength = int64(len(padded))
	return next(req)
}

// padAssistantReasoning rewrites body with reasoning_content added to every
// assistant message that lacks it, returning the original bytes when nothing
// had to change. It fails only on a parse error; the caller then sends the
// original body.
func padAssistantReasoning(body []byte) ([]byte, error) {
	var envelope map[string]jsontext.Value
	if err := json.Unmarshal(body, &envelope); err != nil {
		return nil, err
	}
	rawMessages, ok := envelope["messages"]
	if !ok {
		return body, nil
	}
	var messages []map[string]jsontext.Value
	if err := json.Unmarshal(rawMessages, &messages); err != nil {
		return nil, err
	}
	padded := false
	for _, msg := range messages {
		rawRole, ok := msg["role"]
		if !ok {
			continue
		}
		var role string
		if json.Unmarshal(rawRole, &role) != nil || role != "assistant" {
			continue
		}
		if _, ok := msg["reasoning_content"]; ok {
			continue
		}
		msg["reasoning_content"] = jsontext.Value(`""`)
		padded = true
	}
	if !padded {
		return body, nil
	}
	rawList, err := json.Marshal(messages)
	if err != nil {
		return nil, err
	}
	envelope["messages"] = rawList
	return json.Marshal(envelope)
}
