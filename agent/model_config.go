package agent

import (
	"context"
	"fmt"

	"github.com/coder/acp-go-sdk"
)

// modelConfigOptionID is the id of the model selector this agent offers.
const modelConfigOptionID acp.SessionConfigId = "model"

// Model is a model the agent can run. Ref names it for genkit and is the value
// the model config option carries, so a selection needs no mapping table. Group
// names the provider and must be set: it is the selector's group header.
type Model struct {
	Ref    string
	Name   string
	Group  string
	Config any
}

// SetSessionConfigOption applies a client's change to a session config option.
// ACP answers with the complete option list rather than the change, so the
// response is rebuilt from the session's new state.
func (a *Agent) SetSessionConfigOption(ctx context.Context, params acp.SetSessionConfigOptionRequest) (acp.SetSessionConfigOptionResponse, error) {
	// Only select options are offered; a boolean one would need the client to
	// have advertised support for it first, so it cannot arrive here.
	if params.ValueId == nil {
		return acp.SetSessionConfigOptionResponse{}, fmt.Errorf("unsupported config option type")
	}
	if params.ValueId.ConfigId != modelConfigOptionID {
		return acp.SetSessionConfigOptionResponse{}, fmt.Errorf("unknown config option %q", params.ValueId.ConfigId)
	}
	value := string(params.ValueId.Value)
	if _, ok := a.modelFor(value); !ok {
		return acp.SetSessionConfigOptionResponse{}, fmt.Errorf("unknown model %q", value)
	}
	s, ok := a.get(string(params.ValueId.SessionId))
	if !ok {
		return acp.SetSessionConfigOptionResponse{}, fmt.Errorf("session %s not found", params.ValueId.SessionId)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.model = value
	return acp.SetSessionConfigOptionResponse{ConfigOptions: a.configOptions(s.model)}, nil
}

// configOptions builds the complete config state ACP carries: the whole option
// list with its current values, which session/new and session/set_config_option
// both return rather than a bare change.
func (a *Agent) configOptions(current string) []acp.SessionConfigOption {
	if len(a.models) == 0 {
		return []acp.SessionConfigOption{}
	}
	category := acp.SessionConfigOptionCategoryModel
	return []acp.SessionConfigOption{{
		Select: &acp.SessionConfigOptionSelect{
			Id:           modelConfigOptionID,
			Name:         "Model",
			Category:     &category,
			Type:         "select",
			CurrentValue: acp.SessionConfigValueId(current),
			Options:      a.modelOptions(),
		},
	}}
}

// modelOptions lists the configured models as select values grouped by their
// provider, one group per distinct Group in configuration order.
func (a *Agent) modelOptions() acp.SessionConfigSelectOptions {
	var groups []acp.SessionConfigSelectGroup
	at := map[string]int{}
	for _, m := range a.models {
		value := acp.SessionConfigSelectOption{
			Value: acp.SessionConfigValueId(m.Ref),
			Name:  m.Name,
		}
		if i, ok := at[m.Group]; ok {
			groups[i].Options = append(groups[i].Options, value)
			continue
		}
		at[m.Group] = len(groups)
		groups = append(groups, acp.SessionConfigSelectGroup{
			Group:   acp.SessionConfigGroupId(m.Group),
			Name:    m.Group,
			Options: []acp.SessionConfigSelectOption{value},
		})
	}
	boxed := acp.SessionConfigSelectOptionsGrouped(groups)
	return acp.SessionConfigSelectOptions{Grouped: &boxed}
}

// modelFor returns the configured model with that ref.
func (a *Agent) modelFor(ref string) (Model, bool) {
	for _, m := range a.models {
		if m.Ref == ref {
			return m, true
		}
	}
	return Model{}, false
}
