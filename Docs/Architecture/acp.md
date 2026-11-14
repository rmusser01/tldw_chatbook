# ACP

This document describes the ACP destination: what it actually is (a local Agent-Client-Protocol runtime launcher and session-readiness surface — **not** a wire-protocol client), the process manager, the Console-follow handoff, and its honest boundaries.

## What ACP is here

ACP in this app is a 3-pane destination workbench for launching and inspecting one ACP-compatible local runtime process, creating Console-followable session payloads, and exposing runtime readiness to the Console and Home. `ACPRuntimeProcessManager` spawns the configured command with `shell=False` and stdio redirected to DEVNULL, then **synthesizes** a session payload — there is no JSON-RPC/ACP protocol handling in the codebase. An empty `command` in config keeps the destination "honestly blocked".

## Authoritative files

| File | Role |
| --- | --- |
| `UI/Screens/acp_screen.py` | `ACPScreen(BaseAppScreen)` — list/detail/inspector panes; launch/restart/stop buttons; `#acp-follow-in-console` |
| `ACP_Interop/runtime_process.py` | `ACPRuntimeProcessConfig.from_mapping` (parses `[acp.runtime]`), `ACPRuntimeProcessManager` (`start_session`, `stop`, `snapshot`), `ACPRuntimeProcessStatus` (not_configured/configured/starting/running/failed/stopped) |
| `ACP_Interop/runtime_session.py` | frozen `ACPRuntimeSessionState`; record ids `local:acp_session:<id>`; `to_console_live_work_launch()` (None unless a payload exists) |
| `Chat/console_live_work.py` | `ConsoleLiveWorkSourceReadinessState.from_acp_runtime_status` — maps status to the Console readiness row (Blocked/Starting/Connected/Ready/Failed) |
| `UI/Navigation/pending_handoff_store.py` | `HandoffChannel.ACP_SESSION_TARGET` — the canonical `local:acp_session:` handoff value |
| `app.py` | manager construction at app init; `get_acp_runtime_session_state()`; `open_console_for_live_work`; the Console→ACP backhand staging |

## Session flow (dataflow)

1. **Config**: `[acp.runtime]` (`command`, `args`, `cwd`, `env`, `runtime_id`/`runtime_label`, `startup_timeout_seconds` — default 2 s, floor 0.05). Empty command ⇒ NOT_CONFIGURED; the destination shows setup steps instead of pretending readiness.
2. **Launch**: the screen's launch worker → `manager.start_session(title)`: stop any live process first; `Popen([command, *args], cwd=validated_dir, env=os.environ + config env, stdio → DEVNULL, shell=False)`; poll for early exit until the startup timeout; RUNNING ⇒ mint `session_id` (uuid4 hex) and a payload `{pid, command, args, cwd, started_at}`.
3. **State**: every surface reads through `get_acp_runtime_session_state()` → `ACPRuntimeSessionState.from_any`. Crash detection is **lazy** — the process is only re-polled on snapshot reads; a died process flips to FAILED with "exited with code N".
4. **Console follow**: `#acp-follow-in-console` → `to_console_live_work_launch()` (None without a payload ⇒ warning toast) → stages a Console live-work handoff and navigates to the Console.
5. **Console → ACP backhand**: a Console live-work action targeting ACP stages the `ACP_SESSION_TARGET` handoff (the canonical record id — anything else raises and becomes a warning toast); the ACP screen claims it on mount, matches the current session record, focuses the session row, and acknowledges.
6. **Home**: consumes a boolean (`acp_ready = status == running`) for the dashboard chip.

## Boundaries

- ACP owns runtime launch/setup; the Console consumes session payloads read-only; Home consumes a boolean.
- ACP does **not** expose the agent externally and shares no plumbing with the MCP hub beyond sitting beside it in the Console readiness card (only ACP and MCP may say "Connected", and only from real inputs — TASK-24601).
- Diffs and terminals are explicitly not implemented — the screen renders static "not supported by current runtime payload" copy.
- Session payloads are in-memory only; nothing persists ACP session state across restarts.
- The **server** ACP namespace (`tldw_api/client.py::call_server_acp_endpoint`, `/api/v1/acp/...`) is a separate seam with no production callers today — the local screen/manager is the live surface.

## Failure behaviors

| Case | Behavior |
| --- | --- |
| Not configured | Launch/stop disabled with reasons and setup-step copy |
| Manager missing/attr gaps | Duck-typed access synthesizes a blocked snapshot with recovery text; the screen never crashes |
| Start `OSError` / early exit | FAILED with the exact cause ("could not start: …", "exited before it became ready with code N") |
| Stop timeout | terminate → wait → kill → wait; a kill survivor reports FAILED "did not stop after kill timeout" |
| Snapshot raise in Console tick | The readiness row keeps its last-known text — "never borrow a readiness word it did not measure" |

## Config keys

`[acp.runtime]`: `command` (empty = not configured), `args` (list or shlex string), `cwd` (must exist and be a directory), `env`, `runtime_id` (default `local-acp-runtime`), `runtime_label`, `runtime_version`, `startup_timeout_seconds`. The `acp-defaults` settings category is a contract placeholder — no ACP settings panel exists yet.

## Governing decisions and docs

ADR-015 (`015-shell-destination-ia.md`) seats ACP between MCP and Lab. Tasks: `task-11.5` (ACP runtime session contract — "session readiness explicit, ownership stays in ACP"), `task-60.4.1`, `task-646` (destination handoff ownership + ACP target recovery). User guide: `Docs/User_Guide/acp.md` (a stub). Tests: `Tests/ACP/test_runtime_process.py`, `Tests/tldw_api/test_acp_client.py`.

## Verified gotchas

1. `ACPScreen.on_mount` deliberately does not call `super().on_mount()` — the dispatcher already invokes `BaseAppScreen.on_mount` separately; calling it again would double-run.
2. `runtime_configured` is true whenever id/label are non-empty (i.e., for any parsed section even with an empty command) — the process **status** is the honest signal; config `is_configured` means `bool(command)`.
3. The handoff value must be the exact canonical record id `local:acp_session:<session_id>`.
