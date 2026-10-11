package agent

import (
	"context"
	"os"
	"slices"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/middleware"
)

// turnFilesystem is the agent's file tools for one turn: genkit's
// [middleware.Filesystem], wrapped so the per-call state its New builds —
// the open root, the read-before-write file-state cache, and the queue
// read_file injects file contents through — survives every Generate the
// turn runs.
//
// The wrap exists because that state has to outlive a single Generate: the
// approval gate holds each tool call before it runs, and genkit calls New
// again for the resume call that finally runs it. Memoizing the hooks is
// what keeps an edit_file approved after its read_file from failing with
// "has not been read yet", since the read ran in an earlier Generate of
// the same turn. One turn owns one instance, and the turn lock keeps two
// Generates of a turn from overlapping, so the hooks need no other
// coordination than the mutex below.
type turnFilesystem struct {
	// fs is what genkit builds the hooks from: the directory every path is
	// confined to by os.Root, and write access, since the agent offers
	// write_file and edit_file alongside the read-only tools.
	fs middleware.Filesystem

	mu      sync.Mutex
	hooks   *ai.Hooks     // built on the first New, reused for the rest of the turn
	pending []*ai.Message // injections a resume dropped, waiting to be re-added
}

// newTurnFilesystem points the middleware at the session's working
// directory: the base the tool-call display resolves paths against and the
// root the model's file tools may not escape. A session opened without one
// falls back to the process directory.
func newTurnFilesystem(cwd string) *turnFilesystem {
	if cwd == "" {
		if wd, err := os.Getwd(); err == nil {
			cwd = wd
		}
	}
	return &turnFilesystem{fs: middleware.Filesystem{
		RootDir:          cwd,
		AllowWriteAccess: true,
	}}
}

// Name implements [ai.Middleware], reporting the underlying middleware's own
// name so the hooks log under it.
func (f *turnFilesystem) Name() string { return f.fs.Name() }

// New implements [ai.Middleware]. genkit calls it once per Generate — the
// boundary the memoization deliberately crosses: the first call of the turn
// builds the bundle, every approval resume after it gets the same one back,
// file-state cache and injection queue included.
func (f *turnFilesystem) New(ctx context.Context) (*ai.Hooks, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.hooks != nil {
		return f.hooks, nil
	}
	built, err := f.fs.New(ctx)
	if err != nil {
		return nil, err
	}
	inner := built.WrapGenerate
	hooks := *built
	hooks.WrapGenerate = func(ctx context.Context, params *ai.GenerateParams, next ai.GenerateNext) (*ai.ModelResponse, error) {
		return f.wrapGenerate(ctx, params, inner, next)
	}
	f.hooks = &hooks
	return f.hooks, nil
}

// wrapGenerate brackets the middleware's own WrapGenerate with the
// bookkeeping memoization needs on either side of it.
//
// On the way in it re-adds injections a previous resume dropped, but only
// into a request that will survive: genkit's turn-0 resume branch rebuilds
// its request from the Generate's original options, so anything appended to
// params.Request on that iteration is discarded before the model sees it.
// On the way through it captures exactly those messages — what the
// middleware appended after this hook's entry point, on an iteration that
// discards them — so the flush above can deliver them at the next
// iteration, where the middleware itself would have put them.
func (f *turnFilesystem) wrapGenerate(
	ctx context.Context,
	params *ai.GenerateParams,
	inner func(context.Context, *ai.GenerateParams, ai.GenerateNext) (*ai.ModelResponse, error),
	next ai.GenerateNext,
) (*ai.ModelResponse, error) {
	f.mu.Lock()
	drop := dropsInjectedMessages(params)
	if len(f.pending) > 0 && !drop {
		for _, m := range f.pending {
			if !slices.Contains(params.Request.Messages, m) {
				params.Request.Messages = append(params.Request.Messages, m)
			}
		}
		f.pending = nil
	}
	before := len(params.Request.Messages)
	f.mu.Unlock()

	return inner(ctx, params, func(ctx context.Context, p *ai.GenerateParams) (*ai.ModelResponse, error) {
		if drop && len(p.Request.Messages) > before {
			f.mu.Lock()
			f.pending = append(f.pending, p.Request.Messages[before:]...)
			f.mu.Unlock()
		}
		return next(ctx, p)
	})
}

// dropsInjectedMessages reports whether this iteration is the one genkit
// resolves resume directives on — iteration 0 of a call that carries
// restarts or responses — and so rebuilds its request from the Generate's
// original options rather than the one the WrapGenerate hooks were handed.
// The condition mirrors genkit's own turn-0 check in ai/generate.go: if
// that check changes, this must change with it, or the bookkeeping above
// either double-injects (genkit kept the messages) or loses them (genkit
// dropped what this never predicted).
func dropsInjectedMessages(p *ai.GenerateParams) bool {
	if p.Iteration != 0 || p.Options == nil || p.Options.Resume == nil {
		return false
	}
	return len(p.Options.Resume.Restart) > 0 || len(p.Options.Resume.Respond) > 0
}
