# Console run hooks

The Console supports the six legacy argv-command events and the implemented v2 runtime with structured effects and MCP-backed handlers. Legacy behavior follows ADR-148/197; v2 lifecycle and authority contracts follow ADR-163.

## Legacy authoritative files

| File | Role |
| --- | --- |
| `Agents/run_hooks.py` | The engine: `HOOK_EVENTS`, `HookSpec`, `RunHooksConfig`, `load_hooks_config()`, `RunHooksEngine` (`fire`, `fire_async`, `notify`, `wrap_review`, `post_tool_dep`, `close`) |
| `Chat/console_runtime.py` | `ConsoleRuntime.ensure_run_hooks()` — one engine per app lifetime; closed at the terminal shutdown fence |
| `Chat/console_chat_controller.py` | Fire sites: `UserPromptSubmit` on manual submit, `Stop` on terminal run-state transition, `ApprovalRequested` when a permission round is admitted |
| `Chat/console_agent_bridge.py` | Fire sites: `PreToolUse` via `wrap_review` (guard), `PostToolUse` via the dispatch-loop dependency, `SubagentStop` when a child settles |

Decisions: `backlog/decisions/148-console-run-hooks.md`, `197-console-hook-configuration-review.md`, `163-expanded-console-hook-runtime.md` (the status line predates the live v2 wiring; the modules below implement that contract). Specs: `Docs/superpowers/specs/2026-09-11-console-run-hooks-design.md`, `2026-09-27-console-hook-settings-and-review-design.md`.

## Legacy configuration

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

## The six legacy lifecycle events

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

## Legacy execution mechanics

- **Payload**: JSON on stdin — `{hook_event, session_id, run_id, timestamp, cwd, data}`. `cwd` defaults to `[console] workspace_root`, else the app cwd, overridable per fire. Tool arguments are summarized for notification events (never `str()`-coerced); guard events receive arguments verbatim.
- **Process model**: `subprocess_exec` argv with `start_new_session=True` (process group); timeout kills the whole group (`killpg` SIGKILL; `taskkill /F /T` on Windows), then a bounded 5 s reap. Escaped descendants are outside the guarantee.
- **Capture**: both pipes drained continuously with bounded prefixes (~4000 chars per stream); truncation markers appended. PreToolUse stdout overflow itself denies — incomplete decision JSON must never look like a clean pass.
- **Concurrency**: a dedicated 4-worker pool for blocking events (never shared), a 1-worker notify coordinator plus a separate notification pool; first-deny-wins reduce (remaining processes run out and are discarded). Notify admission is bounded (64 slots; oversized payloads dropped whole with diagnostics).
- **Logging**: every execution is a structured loguru record with session/run ids and a SHA-256 command fingerprint.
- **Shutdown**: `close()` seals admission; live process owners kill groups within 50 ms; a `PreToolUse` fired against a closed engine fails closed.

## Consent layer (ADR-197)

Every enabled hook requires explicit user consent before the next manual Send (background submissions require consent without UI). Grants are scoped to (config file, hook identity, a versioned fingerprint of event+argv+matcher+timeout) and stored in a private atomic local JSON store owned by the shared runtime. Consent failure is restrictive — it cannot inherit `UserPromptSubmit`'s fail-open behavior. Revocation seals current-runtime admission before persistence. The live v2 runtime and managed-plugin hook contributions have separate ownership and validation; legacy grants do not widen v2 authority.

## Implemented v2 runtime (ADR-163)

`Agents/hooks_v2/` owns normalized definitions/results, validation, matching, budgets, command and MCP executors, tool phases, lifecycle checkpoints, context and continuations. `ConsoleRuntime.ensure_hooks_v2()` owns session engines; `prepare_hooks_v2()` binds a live conversation/workspace hook set and completes `SessionStart` before dependent work. The controller wires send and compaction boundaries; the bridge supplies the engine and lifecycle to `AgentService`. Tab focus does not initialize a new session.

In addition to the legacy event names, v2 handles `SessionStart`, `SessionEnd`, `SubagentStart`, `PostToolUseFailure`, `PreCompact`, `PostCompact` and `Interrupt`. Accepted effects are event-specific: context, deny, validated tool-input proposals, narrowing child limits and bounded Stop continuations. Handlers declare `required` and `require_context`; host-owned checkpoint stores accept current effects and release dependent work together. Failed required events block subsequent input/use while retaining settled results. Revocation and owner retirement suppress late effects.

Tool phases transform input, validate final arguments, then apply normal permission review and fresh dispatch checks. MCP-backed hooks retain ordinary tool schemas, approvals, cancellation and causal-cycle checks. A hook cannot bypass permission authority. Stop continuations use bounded scheduler ownership rather than recursive execution. V2 output is parsed through its closed protocol; legacy stdout is never inferred to be v2.

## Platform qualification

V2 command hooks refuse native Windows with `unsupported_platform` before root
reservation, environment access or subprocess launch (ADR-163). Only tests that
need actual command execution require the POSIX backend; validation, unavailable
handlers, pre-dispatch refusals, empty-engine recovery and typed preparation
controls remain active on Windows. The native refusal control also checks repeated
refusal on the same engine and absence of retained process records or budget leaks.

`.github/workflows/tool-recovery.yml` runs native/fenced schema reload recovery,
skill gates, working-set replacement, the full tool pipeline and real bounded
command/cleanup controls on Linux/Python 3.12, retaining JUnit evidence. This is
Linux runtime qualification; it does not reproduce the reporter's exact Fedora
configuration or establish the original run's undisclosed discovery history.

The [2026-10-28 Linux run](https://github.com/rmusser01/tldw_chatbook/actions/runs/40742429112) passed all 142 selected cases with zero skips,
failures or errors on Ubuntu 24.04 / Python 3.12.15. Native Windows qualification
passed 72 cases with 20 explicit POSIX-command skips and no failure or teardown
error; Windows v2 commands remain refused. See TASK-37204 and ADR-163.

## Legacy dataflow

1. `load_hooks_config` parses `[hooks]`; `ConsoleRuntime.ensure_run_hooks()` builds one engine (an unconfigured engine returns `None` and does not latch — configuring hooks later in the same app run works).
2. The controller wires the engine at bridge construction: `guard_tool_calls = engine.wrap_review(...)`, `post_tool_call = engine.post_tool_dep(...)`.
3. A manual send fires `UserPromptSubmit` before turn acceptance; a blocking outcome refuses the send.
4. During the agent loop, each tool batch passes the guard (`PreToolUse`) before the permission review; dispatched calls then fire `PostToolUse` from the dispatch loop.
5. Admitted approval rounds publish `ApprovalRequested`; the terminal run-state transition fires the armed one-shot `Stop`; each settled sub-agent fires `SubagentStop`.

## Related docs

- [console.md](./console.md) — the send lifecycle these events hook into
- [mcp-hub.md](./mcp-hub.md) — the permission store hooks can deny but never bypass
- `Docs/User_Guide/console/agent-runs-and-tools.md` — user-facing hook configuration
