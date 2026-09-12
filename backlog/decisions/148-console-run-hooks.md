# ADR-148: Console run hooks — user-scope external command hooks on session lifecycle events

Status: Proposed
Date: 2026-09-11
Related Spec: [Console run hooks design](../../Docs/superpowers/specs/2026-09-11-console-run-hooks-design.md)
Related: [ADR-069](069-console-project-instruction-local-state-and-preflight.md) (untrusted project context never grants execution)

## Context

The Console had no Claude Code–style hook system: no user-configurable commands at lifecycle points, no `[hooks]` config surface, and the only seams were internal (`CONSOLE_VIEW_HOOK_SLOTS` view binding, `AgentService`'s `review_tool_calls`/`on_step`/`on_child_settled` callbacks). Developers integrating with a console chat session had no sanctioned way to gate tool calls, observe runs, or inject context. Meanwhile every Claude Code event already has a natural attach point in this codebase (review chain, per-step consumer, settle hooks, submit path, approval bridges), and the repo already has a trust-gated script-execution precedent (`run_skill_script`).

## Decision

- **Six v1 events**, Claude Code names where semantics match: `UserPromptSubmit` (manual-origin sends only — the wake invariant), `PreToolUse` (blocking), `PostToolUse`, `ApprovalRequested`, `Stop`, `SubagentStop` (all fire-and-forget).
- **External commands, argv-only, no shell.** A new pure engine module `Agents/run_hooks.py` parses `[hooks]` config, builds JSON payloads, executes via subprocess with per-hook timeouts, truncates output, and logs every execution to the run log.
- **Deny-only guardrails, failing per purpose.** `PreToolUse` hooks can deny a call (which then flows through the existing verdict mechanism as the tool result); they can never allow-bypass the MCP permission store or the builtin gate. `PreToolUse` denies on crash/timeout/unclean exit (fail-closed — a broken guardrail must not open the gate), while `UserPromptSubmit` fails open (a broken hook must not brick the composer); only an explicit deny blocks a send. stdout JSON decisions take precedence over exit-code shorthand.
- **User-scope config only** (`config.toml [hooks]` + `[[hooks.hook]]`). Project-shipped hook files are rejected for v1: ADR-069's rule that project context never grants execution applies squarely to repo-shipped commands that would run on clone.
- **Minimal runtime surface.** One new optional dep `post_tool_call` on the agent runtime deps (the established `run_skill_script` pattern), fired at the dispatch-loop capture point where the uncapped result exists — the step stream cannot serve `PostToolUse` because `AgentStep` results are capped to 2 000 chars and carry no args. Everything else fires from Console-layer seams (controller submit path and review-hook wrapper, bridge terminal-state/`on_child_settled` consumers, approval bridges). The engine is a `ConsoleRuntime`-owned singleton so headless wake runs share it.
- **Thread-model aware execution.** `submit_draft` is async on the event loop, so `UserPromptSubmit` hooks run via an async entry (`asyncio.to_thread`); the run/review chain executes in threads and uses the synchronous entry. Matching hooks run concurrently with first-deny-wins; timeouts kill the hook's whole process group (`start_new_session` + `killpg`; `taskkill /T` on Windows) so a timed-out script's children cannot survive.

## Alternatives

- **In-process Python callback registry** was rejected: usable only by code shipped with the app, no parity with the Claude Code model users asked for, and it introduces a plugin-packaging surface this repo doesn't have.
- **Event-bus taps (observability only)** was rejected: Textual message subscribers can observe but cannot gate or inject, which drops two of the three purposes.
- **Full Claude Code parity in v1** (project scope, decision-JSON input rewriting, SessionStart/End/PreCompact) was rejected as scope creep: each deferred piece needs its own trust or seam design.
- **Shell command strings** were rejected for argv lists: a shell string is an injection surface and adds quoting ambiguity for no capability gain.

## Consequences

- Hooks run with the user's privileges by design; the mitigations are scope (user config only), deny-only verdicts, timeouts with process-group kill, payload redaction/truncation, and full run-log visibility of every execution and every injected context block.
- Invalid hook config fails loud at load (that hook disabled + logged), never a silent no-op.
- Non-blocking events run on a single-worker executor inside the engine and are dropped (logged) under pressure — chat latency is never hostage to hook scripts.
- `UserPromptSubmit` stdout injection gives hooks a turn-scoped context channel; the send can be rejected with exit 2, surfaced as a refusal.
- Future project-scope hooks require a new ADR extending the ADR-069 trust gate; this ADR's deny-only and fail-closed rules are the floor any extension stands on.
