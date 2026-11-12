# Console run hooks

The Console supports the six legacy argv-command events and the implemented v2 runtime with structured effects and MCP-backed handlers. Legacy behavior follows ADR-345/197; v2 lifecycle and authority contracts follow ADR-163.

## Legacy authoritative files

| File | Role |
| --- | --- |
| `Agents/run_hooks.py` | The engine: `HOOK_EVENTS`, `HookSpec`, `RunHooksConfig`, `load_hooks_config()`, `RunHooksEngine` (`fire`, `fire_async`, `notify`, `wrap_review`, `post_tool_dep`, `close`) |
| `Chat/console_runtime.py` | `ConsoleRuntime.ensure_run_hooks()` — one engine per app lifetime; closed at the terminal shutdown fence |
| `Chat/console_chat_controller.py` | Fire sites: `UserPromptSubmit` on manual submit, `Stop` on terminal run-state transition, `ApprovalRequested` when a permission round is admitted |
| `Chat/console_agent_bridge.py` | Fire sites: `PreToolUse` via `wrap_review` (guard), `PostToolUse` via the dispatch-loop dependency, `SubagentStop` when a child settles |

Decisions: `backlog/decisions/345-console-run-hooks.md`, `197-console-hook-configuration-review.md`, `163-expanded-console-hook-runtime.md` (the status line predates the live v2 wiring; the modules below implement that contract). Specs: `Docs/superpowers/specs/2026-09-11-console-run-hooks-design.md`, `2026-09-27-console-hook-settings-and-review-design.md`.

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

Invalid entries fail loud: a warning is logged and that hook is disabled; a non-bool `enabled` disables all hooks. Project-shipped hooks are rejected outright (ADR-317 trust rule — untrusted project files cannot configure hooks).

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

`ConsoleRuntime.ensure_run_hooks()` also attaches the private original source in
`Agents/legacy_hook_ownership.py`. Construction is metadata-only. A selected app
event acquires one complete opaque WorktreeActivity claim and a genuine non-child
ExecutionOwner, reserving each real target's tool operation before the existing
pool/notification enqueue. Capacity/activity refusal preserves the event's
existing policy. No-target and disabled events allocate neither. Standalone
RunHooksEngine callers remain independent of native attachment.

Targets retain their operation through actual callback return, original process
exit plus managed-group absence, and the original host handoff. First-deny keeps
sibling workers owned. Notification discard callbacks are part of that join.
UserPromptSubmit uses the actual submit coroutine, carrying its same handoff into
an existing durable postcommit continuation when necessary. Failed optional audit
publication retains that continuation for retry or conclusive discard; ordinary
required native-write failure and unproven cleanup retain opaque protection.
No hook journal or AgentRun is created, and ephemeral/no-transcript behavior is
unchanged. A recoverable pending publication can therefore block competing opaque
work until retry/discard. Runtime shutdown permanently retires non-resumable host
continuations and joins original callbacks before disposing storage, even when a
shutdown waiter is cancelled. An expired process/group proof remains explicitly
unresolved; no new poller or executor claims later success.
Native attachment requires the metadata-only `native_workflow_qualified()` POSIX
group capability in `Chat/console_workflow_runs.py`. Unqualified platforms,
including Windows, retain ordinary granted app hooks and existing shutdown limits
without native attachment. The same predicate refuses actual workflow authority
before phase acceptance, regardless of the workflow feature flag. Passive saved
workflow history/storage stays available. Future public planning, Start and Resume
must reuse it before billable preparation; Task 9 will surface the reason. Policy
simulation tests do not qualify Windows physical cleanup.

This is Task 8D2B1's legacy boundary. The v2 integration below extends finite
event custody through native command and MCP workers; ordinary covering-tree
borrowing remains mandatory before outer execution is enabled. Workflow capture/planning/coordinator admission remains closed. Runtime
coordination does not exclude unrelated ordinary orphaned effects across restart.

## Native plugin hook preparation (Task 8D2B2a)

Before an engine exists, `ConsoleRuntime.prepare_hooks_v2()` calls the existing
`PluginService.hook_configuration()` with the captured selected component maximum.
Its `component_snapshots()`/`admit()` callbacks perform real plugin admission.
`console_plugin_hook_preparation.NativePluginHookPreparation` binds the original
`_HookPreparationSource`, maximum, published plugin snapshots and original service
infrastructure to one finite opaque activity claim and one non-child tool operation.
The original validated submission supplies manual/automatic origin; labels do not.
No selected plugin rows means no native worker or activity acquisition.

The source is rechecked before native reservation, at service queue publication,
before each actual callback and around the original `NativeHooks` host handoff.
`PluginService._call()` retains and joins the original service Future through
repeated cancellation; its return does not retire the persistent PluginRuntimeOwner.
The existing Console preparation-read observers retain the request through the
actual native release callback. Runtime/session sealing precedes capacity close;
their existing drains wait for those callbacks before disposing shared storage.
The shared `native_workflow_qualified()` predicate preserves the existing ordinary
path on unqualified platforms; it provides no Windows workflow qualification.

Preparation retires before the event owner below begins. Outer ordinary-tree
borrowing remains pending; workflow capture/planning/coordinator admission stays closed.

## Native v2 command and MCP events (Task 8D2B2b1/b2)

`hooks_v2.native_commands.NativeCommandEvents` pins the actual runtime, engine,
session incarnation/binding, permissions, published NativeHooks and service
infrastructure. `HookEngine.begin_event()` freezes the complete event definition
maximum and matcher values. `MCPHookSelection` pins the original context registration,
registry definition, provider entry/maximum, service infrastructure and owned plugin
snapshot. Replacement, revocation or definition widening refuses covered dispatch.
Nested events inherit only the live original delivery, ToolHookRun definition and
private causal chain. Event IDs and input labels cannot select coverage or a fallback.
Accepted context may outlive its spent input registration only after actual host
custody retirement and the existing native owner's positive drain observation.
It still checks the original provider/schema/catalog, plugin/permission and host
sources. Every admission/start/currentness/nested borrow requires original live
registration; closed scopes and returned operations never authorize dispatch.

Before sync/async individual or notification queue publication, a covered event
acquires one opaque claim/non-child owner and reserves its complete reachable worker
union. Metadata-only traversal captures PreToolUse, PostToolUse and failure candidates
under existing matching, dependency/cycle and capacity bounds. Selected DENY, disabled
and impossible branches are excluded; ASK remains subject to ordinary review. Input
and result transformations cannot widen immutable tool/provider matcher metadata.
The actual engine lock rechecks the concrete source before publishing each delivery.
Original permissions, activation guards, plugin admission and hook budgets still
run. No-target/disabled events remain cold, and standalone/unqualified engines keep
their existing path. Actual manual initialization and UserPromptSubmit carry the
validated submission origin; public event labels grant no reserved capacity.

`CommandExecutor` retains native target callbacks through its original process exit,
managed-group absence and required `HookProcessOwner.settle_process()` return.
Cancelling a job joins original threaded callbacks before consuming their resources;
unknown cleanup and failed settlement retain stopping capacity and opaque protection.
The existing reaper may later establish positive terminal proof. No journal payload
is added. Shared `_EventCustody` joins the original operation-release callback too.

`HookSessionLifecycle` retains host custody through actual publish or discard.
`ToolHookRun` keeps it through transformations and checkpoint acceptance; compaction
keeps it through its separate ContextLedger and provider candidate preparation.
The original Stop proposal consumer retains the same event through its existing
queue acceptance/discard; it releases before later ordinary durable admission.
Stopping during a native commit pauses the exact session's machine queue before
cancelling the commit wrapper, invalidating a pending continuation receipt gate.
Each acceptance checks the original source after host callbacks. Closing the event
scope alone cannot retire these holds. Authorized SessionEnd targets are retained
by the original lifecycle seal while capacity remains open. Runtime/session drains
join actual callbacks and host handoffs through repeated waiter cancellation before
removing the session or disposing its storage.

`MCPHookExecutor` enters each actual thread through the existing worker isolation and
reacquires the exact original activation sources and native owner. The sequential
result validator shares the original operation; distinct nested workers each retain
one. Callback return, bridge completion/settled dispatch, original plugin request
`completed` proof and the existing reaper all precede native release. A failed required
plugin settlement retains custody until the original request is actually settled;
logical timeout/cancellation/result alone cannot clear it. The persistent plugin
runtime is not claimed drained by one request's completion. Never-issued conditional
branches are discarded only when their original producer returns; nested host scopes
join the root through transformations/checkpoint acceptance and actual host return.

The live authorized lifecycle seal retains exact MCP/nested operations before capacity
close, so later nested work consumes those holds without a reserve exception. Existing
configured disposed-runtime policy still refuses MCP SessionEnd; custody does not
extend teardown permission or currentness. Full ordinary-tree borrowing and public
workflow capture/coordinator remain closed. There are no new event journal writes.

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
