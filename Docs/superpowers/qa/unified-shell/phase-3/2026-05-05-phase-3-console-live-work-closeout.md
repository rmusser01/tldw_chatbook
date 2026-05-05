# Phase 3 Console Live-Work Closeout

Date: 2026-05-05
Task: `TASK-3.11`
Parent Task: `TASK-3`
Branch: `codex/unified-shell-phase3-console-closeout`
Base: `origin/dev` at `bbeaf32c`

<!-- PHASE_3_CLOSEOUT_METADATA:BEGIN -->
```json
{
  "closeout_task": "TASK-3.11",
  "parent_task": "TASK-3",
  "decision": "verified",
  "verified_sources": [
    "home-active-work",
    "watchlists-collections",
    "schedules",
    "rag-search",
    "artifacts",
    "workflows"
  ],
  "verified_recovery_sources": [
    "acp",
    "mcp",
    "event-streams"
  ],
  "source_readiness": {
    "home-active-work": "connected",
    "watchlists-collections": "connected",
    "schedules": "connected",
    "rag-search": "connected",
    "artifacts": "connected",
    "workflows": "connected",
    "acp": "not_wired_recoverable",
    "mcp": "not_wired_recoverable",
    "event-streams": "future_work_recoverable"
  },
  "baseline_replay_result": {
    "passed": 87,
    "failed": 1,
    "warnings": 1
  },
  "final_focused_replay_result": {
    "passed": 89,
    "failed": 0,
    "warnings": 1
  },
  "final_broader_replay_result": {
    "passed": 154,
    "failed": 0,
    "warnings": 1
  }
}
```
<!-- PHASE_3_CLOSEOUT_METADATA:END -->

## Current Baseline

Phase 3 is replayed after the merged Console live-work slices:

- `TASK-3.1` - typed Console live-work launch contract.
- `TASK-3.2` - reusable Console live-work status-card seam.
- `TASK-3.3` - Home W+C active-work launch into Console.
- `TASK-3.4` - Console W+C primary action routing into W+C run details.
- `TASK-3.5` - W+C destination Console follow producer.
- `TASK-3.6` - Console source-readiness summary.
- `TASK-3.7` - Schedules active-run and reading-digest Console producers.
- `TASK-3.8` - Search/RAG result Console producer.
- `TASK-3.9` - Artifacts Chatbook Console producer.
- `TASK-3.10` - Workflows active-run Console producer.
- `TASK-3.11` - maturity-gate replay and closeout.

The implemented product boundary is Console launch, status, follow, and recovery for connected local sources. ACP, MCP, and deeper event-stream subscriptions are not implemented as live producers yet; they are verified as visible, honest, recoverable source-readiness states.

## Workflow Matrix

| Workflow | Running-app path | Result | Severity |
| --- | --- | --- | --- |
| Home active-work to Console | Home W+C active-work rows produce a typed `HomeConsoleLaunch` and stage `pending_console_launch`. | Functional; Console renders source, title, status, recovery, action label, and payload. | none |
| W+C destination to Console | W+C uses the active-work adapter to launch the latest active W+C run into Console. | Functional when an actionable W+C run exists; unavailable states remain explicit when no run is available. | none |
| Console W+C follow-through | Console `Open W+C run` primary action stages W+C run context and navigates to `subscriptions`. | Functional; not render-only because click routing updates app state and posts navigation. | none |
| Schedules to Console | Schedules launches active schedule runs or latest local reading-digest output through `open_console_for_live_work`. | Functional for implemented local sources; unavailable states explain missing schedule output. | none |
| Search/RAG to Console | Search/RAG result cards stage selected retrieved evidence into Console. | Functional when a selected result exists; missing Console surface and blocked server-backed paths show recovery copy. | none |
| Artifacts to Console | Artifacts launches the latest local Chatbook artifact into Console. | Functional when a local Chatbook exists; service errors and empty states remain recoverable. | none |
| Workflows to Console | Workflows launches active workflow runs from the adapter context into Console. | Functional when adapter context exists; empty context remains honest and disabled. | none |
| ACP source readiness | Console source readiness shows ACP as `Not wired`. | Verified recovery state; runtime/session launch remains future work. | none |
| MCP source readiness | Console source readiness shows MCP as `Not wired`. | Verified recovery state; live MCP run producer remains future work. | none |
| Event streams | Console status cards render staged status, recovery, and action metadata without subscribing to live event streams. | Verified as future-work recovery boundary; no fake live stream is implied. | none |

## Focused Verification

Commands were run from the repository root using the project virtual environment. Portable command form:

- Broader baseline replay: `python3 -m pytest Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_shells.py Tests/UI/test_home_screen.py Tests/UI/test_search_rag_window.py Tests/UI/test_unified_shell_phase234_maturity_gate.py -q`
- Broader baseline replay result before closeout tracker updates: `151 passed, 2 failed, 3 warnings`
- Broader baseline failures:
  - `Tests/UI/test_unified_shell_phase234_maturity_gate.py::test_phase_two_three_four_closeout_tasks_record_current_parent_status` still expected `TASK-3.11` to remain `To Do`.
  - `Tests/UI/test_destination_shells.py::test_destination_action_buttons_emit_compatibility_routes[library-#library-open-notes-notes]` did not record navigation in the broad sweep.
- Isolated Library compatibility rerun: `python3 -m pytest Tests/UI/test_destination_shells.py::test_destination_action_buttons_emit_compatibility_routes --maxfail=1 -q`
- Isolated Library compatibility rerun result: `8 passed, 1 warning`
- Phase 3 focused baseline replay: `python3 -m pytest Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_home_screen.py Tests/UI/test_search_rag_window.py Tests/UI/test_unified_shell_phase234_maturity_gate.py -q`
- Phase 3 focused baseline result before closeout tracker updates: `87 passed, 1 failed, 1 warning`
- Red closeout contract: `python3 -m pytest Tests/UI/test_unified_shell_phase234_maturity_gate.py::test_phase_three_closeout_doc_records_verified_workflows_and_task_completion -q`
- Red closeout result before evidence existed: `1 failed` with `FileNotFoundError` for this closeout document.
- Final focused replay: `python3 -m pytest Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_home_screen.py Tests/UI/test_search_rag_window.py Tests/UI/test_unified_shell_phase234_maturity_gate.py -q`
- Final focused replay result after closeout updates: `89 passed, 1 warning`
- Final broader replay: `python3 -m pytest Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_shells.py Tests/UI/test_home_screen.py Tests/UI/test_search_rag_window.py Tests/UI/test_unified_shell_phase234_maturity_gate.py -q`
- Final broader replay result after closeout updates: `154 passed, 1 warning`

Warning boundary: the remaining warning is an existing Requests dependency-version warning and is not a Console live-work behavior failure.

## UX Notes

- Console now behaves as the live-work hub for connected local sources instead of a passive tab.
- The pending live-work card shows recognizable status, recovery, payload, and primary action information without requiring recall from the source screen.
- Source readiness preserves beginner orientation by showing which source families are connected and which are not wired.
- Power-user speed is preserved because source screens can still stage context into Console directly, and the Console W+C action routes straight to the relevant run details.

## Defects And Blockers

No Phase 3 closeout blocker remains.

The broad replay surfaced one Library compatibility-route failure that did not reproduce in the isolated compatibility rerun. Because Phase 3 focused Console live-work replay stayed green and Library service adoption belongs to Phase 4, this is tracked as a residual flake risk rather than a Phase 3 blocker.

## Residual Risk

- ACP and MCP live-work producers remain future work; Phase 3 verifies honest source-readiness recovery, not runtime-backed ACP/MCP sessions.
- Console still stages status-card state rather than subscribing to a durable live event stream.
- Full first-time-user and power-user shell replay remains Phase 6 scope.

## Closeout Decision: verified

Phase 3 is verified because Console launch, follow, status, and recovery flows are covered for Home active work, W+C, Schedules, Search/RAG, Artifacts, and Workflows. ACP, MCP, and deeper event streams remain explicitly documented recoverable gaps instead of false live controls.
