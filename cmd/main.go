package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/signal"

	"genkit-code-agent/agent"

	"github.com/coder/acp-go-sdk"
)

const usage = `usage: gca <command>

commands:
  acp    start the agent over stdio using the Agent Client Protocol
  help   show this help
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}

	switch os.Args[1] {
	case "acp":
		runACP()
	case "help", "-h", "--help":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "gca: unknown command %q\n\n%s", os.Args[1], usage)
		os.Exit(2)
	}
}

func runACP() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	ag := agent.New()
	conn := acp.NewAgentSideConnection(ag, os.Stdout, os.Stdin)
	conn.SetLogger(slog.Default())
	ag.SetAgentConnection(conn)

	select {
	case <-conn.Done():
		fmt.Fprintln(os.Stderr, "gca: client disconnected")
	case <-ctx.Done():
	}
}
