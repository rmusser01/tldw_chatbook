# Console run hooks

This document describes the user-configured external-command hook subsystem: the six lifecycle events, deny-only semantics, execution mechanics (argv-only, process-group kill, bounded capture), and the consent layer. Governed by ADR-148 as amended by ADR-197.

## Authoritative files

| File | Role |
| --- | --- |
| `Agents/run_hooks.py` | The engine: `HOOK_EVENTS`, `HookSpec`, `RunHooksConfig`, `load_hooks_config()`, `RunHooksEngine` (`fire`, `fire_async`, `notify`, `wrap_review`, `post_tool_dep`, `close`) |
| `Chat/console_runtime.py` | `ConsoleRuntime.ensure_run_hooks()` — one engine per app lifetime; closed at the terminal shutdown fence |
| `Chat/console_chat_controller.py` | Fire sites: `UserPromptSubmit` on manual submit, `Stop` on terminal run-state transition, `ApprovalRequested` when a permission round is admitted |
| `Chat/console_agent_bridge.py` | Fire sites: `PreToolUse` via `wrap_review` (guard), `PostToolUse` via the dispatch-loop dependency, `SubagentStop` when a child settles |

Decisions: `backlog/decisions/148-console-run-hooks.md`, `197-console-hook-configuration-review.md`, `163-expanded-console-hook-runtime.md` (accepted, implementation pending). Specs: `Docs/superpowers/specs/2026-09-11-console-run-hooks-design.md`, `2026-09-27-console-hook-settings-and-review-design.md`.

## Configuration

Hooks live under `[hooks]` in `config.toml`:

```toml
[hooks]
enabled = true

[[hooks.hook]]
event = "PreToolUse"                          # one of the six events
matcher = "Bash*"                             # glob; only valid on PreToolUse/PostToolUse
command = ["/usr/local/bin/guard.sh", "--strict"]  # argv list — no shell, no string commands
timeout_s = 10.0                              # default 10; must be positive and finite
```

Invalid entries fail loud: a warning is logged and that hook is disabled; a non-bool `enabled` disables all hooks. Project-shipped hooks are rejected outright (ADR-069 trust rule — untrusted project files cannot configure hooks).

## The six lifecycle events

| Event | Kind | Fires | Blocking? | Failure mode |
| --- | --- | --- | --- | --- |
| `UserPromptSubmit` | Send | Manual-origin sends, before acceptance | Yes (async entry) | **Fail-open** |
| `PreToolUse` | Tool | Per tool call, before permission exemptions and approval | Yes (sync, in the review chain's thread) | **Fail-closed** |
| `PostToolUse` | Tool | Only for calls that actually dispatched | No | Best-effort |
| `ApprovalRequested` | Approval | When a permission round is admitted | No | Best-effort |
| `Stop` | Turn | Once on the session's terminal run-state transition | No | Best-effort |
| `SubagentStop` | Fleet | When a sub-agent settles | No | Best-effort |

**Deny-only**: hooks can refuse work but can never bypass or widen the permission store. `PreToolUse` denials are merged back into the review verdicts as `"hook: <reason>"` refusals — namespaced so hook text can never equal the `"proceed"` dispatch sentinel. An `allow`/unknown decision from a hook is ignored and logged.

- `UserPromptSubmit`: exit 0 with no parsed `decision` → captured stdout (truncated) becomes a persisted hook-origin SYSTEM transcript row before the assistant row; `{"decision":"block","reason":…}` or exit 2 → the send is refused, run state BLOCKED; crash/timeout → logged, send proceeds.
- `PreToolUse`: crash, nonzero/non-2 exit, timeout, stdout-capture overflow, or engine error all deny. Engine-level failure returns "hooks engine; failing closed". Denied calls are excluded from the approval card entirely.
- `PostToolUse`: receives the still-uncapped result (truncated to budget here); built on `notify` so a slow hook never stalls dispatch.

## Execution mechanics

- **Payload**: JSON on stdin — `{hook_event, session_id, run_id, timestamp, cwd, data}`. `cwd` defaults to `[console] workspace_root`, else the app cwd, overridable per fire. Tool arguments are summarized for notification events (never `str()`-coerced); guard events receive arguments verbatim.
- **Process model**: `subprocess_exec` argv with `start_new_session=True` (process group); timeout kills the whole group (`killpg` SIGKILL; `taskkill /F /T` on Windows), then a bounded 5 s reap. Escaped descendants are outside the guarantee.
- **Capture**: both pipes drained continuously with bounded prefixes (~4000 chars per stream); truncation markers appended. PreToolUse stdout overflow itself denies — incomplete decision JSON must never look like a clean pass.
- **Concurrency**: a dedicated 4-worker pool for blocking events (never shared), a 1-worker notify coordinator plus a separate notification pool; first-deny-wins reduce (remaining processes run out and are discarded). Notify admission is bounded (64 slots; oversized payloads dropped whole with diagnostics).
- **Logging**: every execution is a structured loguru record with session/run ids and a SHA-256 command fingerprint.
- **Shutdown**: `close()` seals admission; live process owners kill groups within 50 ms; a `PreToolUse` fired against a closed engine fails closed.

## Consent layer (ADR-197)

Every enabled hook requires explicit user consent before the next manual Send (background submissions require consent without UI). Grants are scoped to (config file, hook identity, a versioned fingerprint of event+argv+matcher+timeout) and stored in a private atomic local JSON store owned by the shared runtime. Consent failure is restrictive — it cannot inherit `UserPromptSubmit`'s fail-open behavior. Revocation seals current-runtime admission before persistence. ADR-162 (managed plugins) and ADR-163 (expanded runtime) are outside the implemented scope.

## Dataflow

1. `load_hooks_config` parses `[hooks]`; `ConsoleRuntime.ensure_run_hooks()` builds one engine (an unconfigured engine returns `None` and does not latch — configuring hooks later in the same app run works).
2. The controller wires the engine at bridge construction: `guard_tool_calls = engine.wrap_review(...)`, `post_tool_call = engine.post_tool_dep(...)`.
3. A manual send fires `UserPromptSubmit` before turn acceptance; a blocking outcome refuses the send.
4. During the agent loop, each tool batch passes the guard (`PreToolUse`) before the permission review; dispatched calls then fire `PostToolUse` from the dispatch loop.
5. Admitted approval rounds publish `ApprovalRequested`; the terminal run-state transition fires the armed one-shot `Stop`; each settled sub-agent fires `SubagentStop`.

## Related docs

- [console.md](./console.md) — the send lifecycle these events hook into
- [mcp-hub.md](./mcp-hub.md) — the permission store hooks can deny but never bypass
- `Docs/User_Guide/console/agent-runs-and-tools.md` — user-facing hook configuration
