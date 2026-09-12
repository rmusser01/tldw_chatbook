# ADR-148: Console run hooks — user-scope external command hooks on session lifecycle events

Status: Accepted (2026-09-12)
Date: 2026-09-11
Related Spec: [Console run hooks design](../../Docs/superpowers/specs/2026-09-11-console-run-hooks-design.md)
Related: [ADR-069](069-console-project-instruction-local-state-and-preflight.md) (untrusted project context never grants execution)

## Context

The Console had no Claude Code–style hook system: no user-configurable commands at lifecycle points, no `[hooks]` config surface, and the only seams were internal (`CONSOLE_VIEW_HOOK_SLOTS` view binding, `AgentService`'s `review_tool_calls`/`on_step`/`on_child_settled` callbacks). Developers integrating with a console chat session had no sanctioned way to gate tool calls, observe runs, or inject context. Meanwhile every Claude Code event already has a natural attach point in this codebase (review chain, per-step consumer, settle hooks, submit path, approval bridges), and the repo already has a trust-gated script-execution precedent (`run_skill_script`).

## Decision

- **Six v1 events**, Claude Code names where semantics match: `UserPromptSubmit` (manual-origin sends only — the wake invariant), `PreToolUse` (blocking), `PostToolUse`, `ApprovalRequested`, `Stop`, `SubagentStop` (all fire-and-forget).
- **External commands, argv-only, no shell.** A new pure engine module `Agents/run_hooks.py` parses `[hooks]` config, builds JSON payloads, executes via subprocess with per-hook timeouts, truncates output, and logs every execution as structured loguru records carrying session/run ids (Rulings R13/R27: non-blocking fires also log truncated stdout/stderr; blocking fires log ids + exit — recorded in spec §12).
- **Deny-only guardrails, failing per purpose.** `PreToolUse` hooks can deny a call (which then flows through the existing verdict mechanism as the tool result); they can never allow-bypass the MCP permission store or the builtin gate. `PreToolUse` denies on crash/timeout/unclean exit (fail-closed — a broken guardrail must not open the gate), while `UserPromptSubmit` fails open (a broken hook must not brick the composer); only an explicit deny blocks a send. stdout JSON decisions take precedence over exit-code shorthand.
- **User-scope config only** (`config.toml [hooks]` + `[[hooks.hook]]`). Project-shipped hook files are rejected for v1: ADR-069's rule that project context never grants execution applies squarely to repo-shipped commands that would run on clone.
- **Runtime boundaries.** Optional `guard_tool_calls` applies deny-only restrictions to every call before approval exemptions and permission review. Guard failures deny the batch; call-ID refusals survive downstream review failures. This replaces using the permission-review wrapper as the production guard seam, which missed preauthorized Canvas calls. The optional `post_tool_call` dep captures dispatched results before truncation, including durable continuation runs. Console submission, the shared interrupt host, and settlement callbacks own the other events. `ConsoleRuntime` owns one synchronized engine and closes it on disposal.
- **Thread-model aware execution.** `submit_draft` is async on the event loop, so `UserPromptSubmit` hooks run via an async entry (`asyncio.to_thread`); the run/review chain executes in threads and uses the synchronous entry. Matching hooks run concurrently within bounded worker capacity with first-deny-wins. Timeouts kill the hook's process group (`start_new_session` + `killpg`; `taskkill /T` on Windows); descendants that deliberately escape the group are outside that guarantee.

## Alternatives

- **In-process Python callback registry** was rejected: usable only by code shipped with the app, no parity with the Claude Code model users asked for, and it introduces a plugin-packaging surface this repo doesn't have.
- **Event-bus taps (observability only)** was rejected: Textual message subscribers can observe but cannot gate or inject, which drops two of the three purposes.
- **Full Claude Code parity in v1** (project scope, decision-JSON input rewriting, SessionStart/End/PreCompact) was rejected as scope creep: each deferred piece needs its own trust or seam design.
- **Shell command strings** were rejected for argv lists: a shell string is an injection surface and adds quoting ambiguity for no capability gain.

## Consequences

- Hooks run with the user's privileges by design; the mitigations are scope (user config only), deny-only verdicts, timeouts with process-group kill, payload truncation — tool args pass through verbatim per Ruling R28, already recorded in spec §12 — and execution metadata in application logs. Non-blocking output is logged at the bounded size; injected context has a hook-origin transcript row. `search_run_log` does not expose these application-log records.
- Invalid hook config fails loud at load (that hook disabled + logged), never a silent no-op.
- Non-blocking firings have bounded admission and a separate execution pool, so matching hooks can run concurrently without consuming blocking-hook capacity. Excess or oversized notification events are dropped whole and logged. Tool arguments remain exact for blocking guards; truncating them could hide the content a guard is supposed to deny.
- `UserPromptSubmit` stdout injection gives hooks a turn-scoped context channel; the send can be rejected with exit 2, surfaced as a refusal.
- Future project-scope hooks require a new ADR extending the ADR-069 trust gate; this ADR's deny-only and fail-closed rules are the floor any extension stands on.

## PR #2645 review amendment (2026-09-12)

The rebase review found that permission-review exemptions, durable submission, and
the unified interrupt host had outgrown the original integration seams. The
restriction-only guard is intentionally separate from permission review: routing
Canvas calls back through interactive approval would violate its existing UX
contract. `Stop` observes individual accepted turns, including queue entries,
independently of toast suppression. Prompt refusals/cancellation release their
preparation, and accepted hook context enters both durable and ordinary requests.

The 4,000-character output limit applies during capture, not after an unbounded
`communicate()`. Resource ownership covers active child processes and pending
notifications at runtime shutdown. See TASK-32507 and the review dispositions in
`Docs/superpowers/reviews/2026-09-12-pr-2645-run-hooks.md`.
