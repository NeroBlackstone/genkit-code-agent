package agent

import (
	"github.com/coder/acp-go-sdk"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/middleware"
)

// useSkillToolName is the tool genkit's Skills middleware contributes to load
// one skill's SKILL.md body on demand. It matches genkit's own constant.
const useSkillToolName = "use_skill"

// turnMiddleware builds the middleware for one Generate call, ordered
// outer-to-inner: the gate holds a call before the reporter sees it, and the
// reporter's Ref is on the request the permission prompt names. Rebuilt per
// call so an allow_always answered mid-turn reaches the next Generate, with
// one exception — fs is the turn's [turnFilesystem], built once per turn so
// the file-state a read_file recorded survives until the write or edit that
// follows it is approved. use_skill is pre-approved — it only reads a local
// file. Skills takes genkit's default: a "skills" directory under the
// process working directory, empty when absent.
//
// SoftToolErrors covers every tool: a wrong file path or bad query comes
// back to the model as {"error": ...} for it to correct, rather than ending
// the turn. It leaves the gate's interrupts alone — a call held for
// approval still reaches the client — and the reporter still files each
// error as a failed call, so the client sees the failure the model then
// retries under a new call.
func (a *Agent) turnMiddleware(s *Session, sid acp.SessionId, fs *turnFilesystem) []ai.Middleware {
	return []ai.Middleware{
		&middleware.ToolApproval{AllowedTools: append(s.allowedTools(), useSkillToolName)},
		toolCallReporter{agent: a, sid: sid, cwd: s.cwd},
		fs,
		&middleware.Skills{},
		&middleware.SoftToolErrors{},
	}
}
