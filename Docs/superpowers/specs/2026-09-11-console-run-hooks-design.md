# Console Run Hooks — Design Spec

Date: 2026-09-11
Status: Implemented
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
- Settings-screen UI. Config-file only in v1; a dedicated settings
  sub-screen is **confirmed for the immediately following PR** (user
  decision 2026-09-11) — this spec's config schema is the contract that
  sub-screen will edit.
- Shell command strings. Commands are argv lists only — no shell parsing, no
  injection surface.

## 3. Event catalogue (6 events)

| Event | Fires | Seam (existing code) | Blocking? |
|---|---|---|---|
| `UserPromptSubmit` | after submit gates, before turn compose; **manual-origin sends only** | controller submit path, beside the `prompt_history` slot write | yes (exit 2 rejects the send) |
| `PreToolUse` | per tool-call batch, before permission review | wrapper around `build_tool_review_hook` verdict chain (`Chat/console_chat_controller.py`) | yes (exit 2 denies the batch) |
| `PostToolUse` | after dispatch, at the run-log capture point where the full result exists; **only for calls that actually dispatched** (verdict `proceed` — refusals fire nothing) | new optional `post_tool_call` dep on the runtime deps (same shape as `run_skill_script`), fired from `Agents/agent_runtime.py`'s dispatch loop; the step stream stays out of it — `AgentStep` results are capped to 2 000 chars and carry no args | no |
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
- **Process**: argv list, **no shell**. The engine offers two entries over one
  implementation: `fire()` (synchronous `subprocess`, for thread contexts —
  the run/review chain executes in threads) and `fire_async()`
  (`asyncio.to_thread` around `fire()`, for event-loop contexts).
  **`UserPromptSubmit` must use `fire_async()`**: `submit_draft` is `async`
  and runs on the event loop (its existing hooks fire synchronously "from
  deep inside submit_draft"), so a blocking subprocess there would freeze
  the TUI.
- **Concurrency**: matching hooks for one firing run **concurrently** on a
  bounded pool — worst-case wall time is the slowest hook, not the sum.
  For blocking events the first deny wins and remaining results are
  discarded. Non-blocking events are scheduled on a single-worker executor
  and never delay the turn; a full queue drops the firing with a log line
  (hooks must not stall chat).
- **Timeout kill**: per-hook `timeout_s`, default 10 s. Hooks are started
  with `start_new_session=True` and killed as a **process group**
  (`os.killpg`) so the script's own children cannot survive the timeout;
  Windows falls back to `taskkill /T`.
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

## 5. Verdict semantics

| Event | clean pass | explicit deny | anything else |
|---|---|---|---|
| `UserPromptSubmit` | exit 0, no JSON: stdout (≤ budget) is prepended as turn context | exit 2 or JSON `{"decision":"block"}`: send rejected; reason surfaced to the user as a refusal | other non-zero / crash / timeout: **warning logged, send proceeds** — a broken hook must not brick the composer |
| `PreToolUse` | exit 0, no JSON: no opinion; normal permission flow continues | exit 2 or JSON `{"decision":"deny","reason":"…"}`: deny; reason becomes the tool result the model sees | **deny — fail-closed**, reason "hook `<name>` failed (exit/timeout)"; the failure is visible to model and run log, never silent |
| all others | stdout logged to run log | n/a (no verdict semantics) | logged, ignored |

Precedence rules for blocking events: stdout that parses as a JSON object with
a `decision` key wins over the exit code (exit 2 is shorthand for the event's
blocking decision — `deny` for `PreToolUse`, `block` for `UserPromptSubmit`);
unparseable stdout with exit 0 is a clean pass with a warning;
`{"decision":"allow"}` on `PreToolUse` is parsed, ignored, and logged — the
deny-only stance is enforced in the engine, not by hook authors.

Injected `UserPromptSubmit` context is model-visible text, so it is recorded
in the run log as its own entry (source hook named) — auditable after the
fact, never silently merged into the prompt.

Matching: optional `matcher` is a glob against the tool name
(`fs_*`, `mcp__github__*`). No matcher = fires for every call. `matcher` is
**only valid on `PreToolUse` / `PostToolUse`**; on any other event it is a
config validation error (there is no tool name to match).

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

Validation is fail-loud: unknown event names, `matcher` on a non-tool event,
empty/non-list `command`, non-positive timeouts are config errors (logged,
that hook disabled) — never silent no-ops. Config is re-parsed when the
file's mtime changes (checked per fire, cached between), and the master
`enabled` switch is read fresh per fire — both follow the kill-switch
"read fresh per turn" precedent.

## 7. Payload hygiene

- No secrets: payloads are built from session/run ids, tool names, args and
  results only — never env, config values, or API keys.
- Tool args and results are truncated to the payload budget before the child
  sees them.
- Hook processes inherit the user's privileges by design (that is what a hook
  is); the spec's mitigations are scope (user-config only), deny-only verdicts,
  timeouts, and full logging of every execution to the run log.

## 8. Wiring (what changes where)

The engine is a **`ConsoleRuntime`-owned singleton** (like the controller and
store): one shared executor, survives view detach, reachable from headless
wake runs. One new `AgentService` surface, mirroring an established dep:

1. `Chat/console_chat_controller.py` — call `await engine.fire_async(...)`
   (`UserPromptSubmit`) in the submit path beside the `prompt_history` write,
   **past the gates, still on the event loop** — `fire_async` keeps the loop
   free while hooks run; wrap `build_tool_review_hook`'s verdict callable so
   `PreToolUse` hooks run first and can deny into the existing verdict
   mechanism (that chain executes in a thread — plain `engine.fire`).
2. `Agents/agent_runtime.py` + `Agents/agent_service.py` — a new optional
   `post_tool_call` dep (the `run_skill_script` pattern: optional field,
   guarded call at the site), fired at the dispatch-loop capture point where
   the full, uncapped result exists, only when `verdict == "proceed"`.
3. `Chat/console_agent_bridge.py` — at the run terminal-state path fire
   `Stop`; keep the `on_child_settled` partial and fire `SubagentStop`
   beside it; wire `post_tool_call` through to the service constructor.
4. Approval bridges (`set_pending_approval` / `park_pending_approval` sites) —
   fire `ApprovalRequested` (these can be UI-thread bridges; the engine's
   executor keeps the subprocess off the UI thread).
5. `Agents/run_hooks.py` — new engine module (the only new file).
6. `config.py` — parse/validate `[hooks]`.

## 9. Error handling summary

The fail direction is **per purpose**, not global: a guardrail failing must
not open the gate; a convenience failing must not brick the composer.

- Config invalid → that hook disabled + loud log.
- `PreToolUse`: deny on crash/timeout/non-clean exit (fail-closed), with the
  failure reason as the visible tool result.
- `UserPromptSubmit`: crash/timeout/non-clean exit → warning logged, send
  proceeds (fail-open); only an explicit deny blocks.
- Non-blocking event crash/timeout → logged, dropped.
- Engine never raises into the turn path: every firing is wrapped; internal
  errors degrade to "hook did not run" with the event's failure semantics;
  executor task failures are logged.

## 10. Testing plan

Unit (`Tests/Agents/test_run_hooks.py`):
- config parse/validation matrix (unknown event, matcher on non-tool event,
  bad glob, empty command, timeout bounds, master switch)
- subprocess execution via stub `python -c` commands: exit 0/2, JSON-decision
  precedence over exit code, stdout capture, timeout kill (process group —
  the stub spawns a surviving child that must die), truncation, redaction
- deny-only enforcement (`allow` decision ignored)
- concurrent hooks: first-deny-wins, wall time ≈ slowest hook
- executor drop semantics for non-blocking events
- `fire_async` keeps a running event loop responsive while a hook sleeps

Integration (mirroring `Tests/Chat/test_console_local_review_hook.py`
patterns):
- PreToolUse deny short-circuits before the permission store; allow does not
  bypass it
- PreToolUse fail-closed on crash/timeout, reason visible in the tool result
- UserPromptSubmit: exit-2 rejection; stdout injection (and its run-log
  record); wake notice fires nothing; loop stays responsive during a slow
  hook
- PostToolUse fires with full (engine-truncated) args+result from the
  runtime dep, and fires nothing for refused calls
- Stop/SubagentStop/ApprovalRequested fire with correct payloads from their
  seams (viewless wake path included)

## 11. Future work (explicitly out of v1)

Project-scope hooks behind a trust gate; decision-JSON input rewriting;
SessionStart/SessionEnd; PreCompact (rewind/summarize seam); **dedicated
settings sub-screen (confirmed follow-up PR, editing the config schema
defined here)**; per-session hook overrides.

## 12. Implementation Notes

Recorded at implementation close-out (2026-09-12):

- **Execution logging (Ruling R13).** The spec's §4 "written to the run
  log" phrasing is implemented as structured loguru records: every firing
  logs the event, session/run ids, exit status, timing, and captured
  (truncated) stdout/stderr at INFO. For events fired outside an active
  run this is the observability surface — there is no run-log row to
  write.
- **`Stop` carries `run_id` null in v1 (Ruling R24).** The run-state seam
  where `Stop` fires has no run identity, so the envelope's `run_id` is
  `null` rather than a fabricated id. Correlate a Stop with its run via
  the preceding `PostToolUse`/`SubagentStop` firings for the same
  `session_id`.
- **Injected `UserPromptSubmit` context delivery (Ruling R22).** Clean-exit
  stdout is delivered to the model as a payload-only trailing user-role
  entry (the wake-notice delivery precedent) — never written to the store,
  turn-scoped, gone on the next history rebuild. The auditable record is a
  separate hook-origin SYSTEM transcript row carrying the injected text.
- **Config liveness (Ruling R26).** §6's "re-parsed when the file's mtime
  changes (checked per fire, cached between)" did not ship: the engine
  re-validates the app's *loaded* configuration on every fire, so config
  edits land when settings are reloaded/saved (F9 Settings) or the app
  restarts. The kill-switch intent survives — `enabled = false` plus a
  reload stops every hook on the next fire — but there is no live
  mtime watch on `config.toml`.
- **Tool args pass through verbatim (Ruling R28).** §7's "tool args and
  results are truncated to the payload budget" is half-true as shipped:
  results (and prompts, hook stdout/stderr, and deny reasons) are capped
  by the shared budget, but tool args are delivered to the hook
  untruncated. Args come from the model's own tool-call JSON and a guard
  hook needs the real body (e.g. an `fs_write` content check); the hook
  is user-configured and trusted with it.
