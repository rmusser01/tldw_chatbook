# Console Run Hooks — Design Spec

Date: 2026-09-11
Status: Draft — pending user review
ADR: backlog/decisions/148-console-run-hooks.md
Related: ADR-069 (untrusted project context), `Docs/superpowers/specs/2026-08-20-agents-md-support-design.md`

## 1. Goal

Give developers Claude Code–style lifecycle hooks in a Console chat session:
user-configured external commands that run at defined points of a session/turn
lifecycle, with a JSON protocol on stdin and exit-code semantics on stdout.

v1 serves three purposes:

1. **Guardrails** — inspect and deny tool calls before dispatch (`PreToolUse`).
2. **Automation / notification** — react to lifecycle facts (approval waiting,
   run finished, tool failed) with fire-and-forget commands.
3. **Context injection** — add text context to a turn at submit time
   (`UserPromptSubmit`).

## 2. Non-goals (v1)

- **Project-scoped hooks** (repo-shipped hook config like Claude Code's
  `.claude/settings.json`). Deliberately excluded: ADR-069 treats
  project-provided content as untrusted and it must never grant execution. A
  project hooks file is a drive-by-execution vector. Revisit only behind an
  explicit trust gate (future ADR).
- **Tool-input rewriting** (`updatedInput` / decision-JSON mutation). Hooks may
  deny, never rewrite.
- **`allow` bypass.** A hook verdict can never skip the MCP permission store or
  the builtin tool gate. Hooks only add restrictions.
- SessionStart/SessionEnd/PreCompact events (no clean, well-defined seam yet;
  sessions are lazily created and long-lived).
- Settings-screen UI. Config-file only in v1; the F9 settings surface follows
  later if the feature sticks.
- Shell command strings. Commands are argv lists only — no shell parsing, no
  injection surface.

## 3. Event catalogue (6 events)

| Event | Fires | Seam (existing code) | Blocking? |
|---|---|---|---|
| `UserPromptSubmit` | after submit gates, before turn compose; **manual-origin sends only** | controller submit path, beside the `prompt_history` slot write | yes (exit 2 rejects the send) |
| `PreToolUse` | per tool-call batch, before permission review | wrapper around `build_tool_review_hook` verdict chain (`Chat/console_chat_controller.py`) | yes (exit 2 denies the batch) |
| `PostToolUse` | after dispatch, at tool_result capture | bridge `on_step` consumer on `tool_result` records (no AgentService change) | no |
| `ApprovalRequested` | when an approval round is armed | `set_pending_approval` / `park_pending_approval` bridge sites | no |
| `Stop` | a session turn's run reaching terminal state | bridge run terminal-state path (active and non-active sessions) | no |
| `SubagentStop` | a fleet child run settling | the `on_child_settled` wiring in `Chat/console_agent_bridge.py` | no |

The **wake invariant** holds: wake notices are machine-origin, so
`UserPromptSubmit` never fires for them (same rule the `prompt_history` slot
already documents). `Stop`/`SubagentStop` fire for wake turns too — they report
run outcomes, not user input.

## 4. Execution model

- **Engine module** `tldw_chatbook/Agents/run_hooks.py`: config parsing +
  validation, payload assembly, subprocess execution, redaction/truncation,
  logging. No UI imports; headless-testable.
- **Process**: argv list, executed synchronously via `subprocess`
  (**no shell**) — blocking events run inline in their calling context (submits
  and run dispatch are already workers, never the UI thread); non-blocking
  events run on the engine's executor (below).
- **cwd**: the session's workspace root when one is bound, else the app cwd
  (mirrors `[console] workspace_root` fallback semantics).
- **stdin**: one JSON document — common envelope + event-specific `data`:

```json
{
  "hook_event": "PreToolUse",
  "session_id": "…",
  "run_id": "…",
  "timestamp": "2026-09-11T12:00:00+00:00",
  "cwd": "/workspace/root",
  "data": { "tool_name": "fs_write", "tool_args": { "path": "…" } }
}
```

Event-specific `data` fields:

- `UserPromptSubmit`: `{"prompt": "…truncated…"}`
- `PreToolUse`: `{"tool_name": "fs_write", "tool_args": {…}}`
- `PostToolUse`: `{"tool_name": …, "tool_args": {…}, "tool_result": "…truncated…", "is_error": bool}`
- `ApprovalRequested`: `{"calls": [{"name": …, "args_summary": …}], "session_active": bool}`
- `Stop`: `{"status": "completed|error|cancelled"}`
- `SubagentStop`: `{"child_run_id": …, "status": …}`

A single shared budget constant (default 4 000 chars) governs every truncation
site: payload fields, hook stdout, and hook stderr.

`PreToolUse` matching is **per tool call**: the engine fans a review batch out
per call whose name matches the hook's glob, runs each matching hook once per
call, and merges results into the existing per-call verdict dict. Calls a hook
does not match are untouched.
- **stdout/stderr**: captured (truncated to the shared budget) and written to
  the run log (`search_run_log` surfaces it).
- **Timeout**: per-hook `timeout_s`, default 10 s, hard-killed after.
- **Concurrency**: blocking events (`UserPromptSubmit`, `PreToolUse`) run
  sequentially inline in the calling context (already off the UI thread —
  submits and run dispatch are workers). Non-blocking events are handed to a
  single-worker executor inside the engine and never delay the turn; a full
  executor queue drops the firing with a log line (hooks must not stall chat).

## 5. Verdict semantics

| Event | exit 0 | exit 2 | other non-zero / crash / timeout |
|---|---|---|---|
| `UserPromptSubmit` | stdout (≤ budget) is prepended as context for this turn | send rejected; reason surfaced to the user as a refusal | warning logged; send proceeds |
| `PreToolUse` | no opinion; normal permission flow continues | deny; reason becomes the tool result the model sees | **deny — fail-closed** (repo convention) |
| all others | stdout logged to run log | same as non-zero | logged, ignored |

`PreToolUse` also accepts a JSON stdout body `{"decision": "deny", "reason":
"…"}`. `{"decision": "allow"}` is parsed, ignored, and logged — the deny-only
stance is enforced in the engine, not by hook authors.

Matching: optional `matcher` is a glob against the tool name
(`fs_*`, `mcp__github__*`). No matcher = fires for every call.

## 6. Config surface

`config.toml`, following the existing `[console]` section pattern in
`config.py`:

```toml
[hooks]
enabled = true          # master switch; false disables every firing

[[hooks.hook]]
event = "PreToolUse"    # one of the six names; unknown = validation error
matcher = "fs_*"        # optional, tool-name glob
command = ["/usr/local/bin/guard.sh", "--strict"]   # argv; required, non-empty
timeout_s = 10          # optional, default 10
```

Validation is fail-loud at load: unknown event names, empty/non-list `command`,
non-positive timeouts are config errors (logged, that hook disabled) — never
silent no-ops.

## 7. Payload hygiene

- No secrets: payloads are built from session/run ids, tool names, args and
  results only — never env, config values, or API keys.
- Tool args and results are truncated to the payload budget before the child
  sees them.
- Hook processes inherit the user's privileges by design (that is what a hook
  is); the spec's mitigations are scope (user-config only), deny-only verdicts,
  timeouts, and full logging of every execution to the run log.

## 8. Wiring (what changes where)

No new `AgentService` constructor surface. All fire points are Console-layer,
using seams that already exist:

1. `Chat/console_chat_controller.py` — call `engine.fire(UserPromptSubmit)` in
   the submit path beside the `prompt_history` write; wrap
   `build_tool_review_hook`'s verdict callable so `PreToolUse` hooks run first
   and can deny into the existing verdict mechanism.
2. `Chat/console_agent_bridge.py` — in the `on_step` consumer, fire
   `PostToolUse` for `tool_result` records; at the run terminal-state path fire
   `Stop`; keep the `on_child_settled` partial and fire `SubagentStop` beside
   it.
3. Approval bridges (`set_pending_approval` / `park_pending_approval` sites) —
   fire `ApprovalRequested`.
4. `Agents/run_hooks.py` — new engine module (the only new file).
5. `config.py` — parse/validate `[hooks]`.

## 9. Error handling summary

- Config invalid → that hook disabled + loud log at load.
- PreToolUse crash/timeout → deny (fail-closed), logged.
- Non-blocking event crash/timeout → logged, dropped.
- Engine never raises into the turn path: every firing is wrapped; internal
  errors degrade to "hook did not run" with the event's failure semantics.

## 10. Testing plan

Unit (`Tests/Agents/test_run_hooks.py`):
- config parse/validation matrix (unknown event, bad matcher, empty command,
  timeout bounds, master switch)
- subprocess execution via stub `python -c` commands: exit 0/2, stdout capture,
  timeout kill, truncation, redaction
- deny-only enforcement (`allow` decision ignored)
- executor drop semantics for non-blocking events

Integration (mirroring `Tests/Chat/test_console_local_review_hook.py`
patterns):
- PreToolUse deny short-circuits before the permission store; allow does not
  bypass it
- PreToolUse fail-closed on crash/timeout
- UserPromptSubmit: exit-2 rejection; stdout injection; wake notice fires
  nothing
- PostToolUse/Stop/SubagentStop/ApprovalRequested fire with correct payloads
  from their seams (viewless wake path included)

## 11. Future work (explicitly out of v1)

Project-scope hooks behind a trust gate; decision-JSON input rewriting;
SessionStart/SessionEnd; PreCompact (rewind/summarize seam); settings-screen
surface; per-session hook overrides.
