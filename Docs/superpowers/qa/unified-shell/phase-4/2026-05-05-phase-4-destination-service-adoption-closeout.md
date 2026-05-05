# Phase 4 Destination Service-Adoption Closeout

Date: 2026-05-05
Task: `TASK-5.6`
Parent Task: `TASK-5`
Branch: `codex/unified-shell-phase4-destination-closeout`
Base: `origin/dev` at `4e025bf9`

<!-- PHASE_4_CLOSEOUT_METADATA:BEGIN -->
```json
{
  "closeout_task": "TASK-5.6",
  "parent_task": "TASK-5",
  "decision": "verified",
  "verified_destinations": [
    "mcp",
    "skills",
    "library",
    "personas",
    "watchlists-collections",
    "schedules",
    "workflows",
    "artifacts",
    "settings"
  ],
  "verified_recovery_destinations": [
    "acp"
  ],
  "destination_readiness": {
    "mcp": "connected",
    "skills": "connected",
    "library": "connected",
    "personas": "connected",
    "watchlists-collections": "connected",
    "schedules": "connected_or_recoverable",
    "workflows": "connected_or_recoverable",
    "artifacts": "connected_or_recoverable",
    "settings": "connected",
    "acp": "runtime_not_configured_recoverable"
  },
  "baseline_replay_result": {
    "passed": 125,
    "failed": 1,
    "warnings": 3
  },
  "red_closeout_contract_result": {
    "passed": 0,
    "failed": 1,
    "warnings": 0
  },
  "final_focused_replay_result": {
    "passed": 5,
    "failed": 0,
    "warnings": 0
  },
  "final_broader_replay_result": {
    "passed": 127,
    "failed": 0,
    "warnings": 1
  }
}
```
<!-- PHASE_4_CLOSEOUT_METADATA:END -->

## Current Baseline

Phase 4 is replayed after the merged destination service-adoption slices:

- `TASK-5.1` - top-level MCP embeds the existing Unified MCP management panel.
- `TASK-5.2` - Skills lists local Agent Skills through `skills_scope_service` and stages concrete Skills context into Console.
- `TASK-5.3` - Library lists local notes, media, and conversations through source services and stages concrete Library context into Console.
- `TASK-5.4` - Personas lists local characters and persona profiles through `character_persona_scope_service` and stages concrete behavior context into Console.
- `TASK-5.5` - W+C lists local monitored sources and saved collection items through Watchlists and Media Reading services and stages concrete W+C context into Console.
- `TASK-5.6` - maturity-gate replay and closeout.

The implemented product boundary is destination service adoption, not full CRUD or server parity for every module. Destinations with available local services now expose concrete list, route, or Console-staging workflows. Destinations without configured runtimes keep honest disabled recovery states instead of fake launch controls.

## Workflow Matrix

| Destination | Running-app path | Result | Severity |
| --- | --- | --- | --- |
| MCP | Open MCP or legacy `tools_settings`; `MCPScreen` mounts `UnifiedMCPPanel` with source, server, scope, section, action, payload, and run controls. | Functional destination adoption; MCP no longer hides behind a placeholder or requires users to know the legacy route. | none |
| Skills | Open Skills with a local `skills_scope_service`; listed skills render counts, names, descriptions, and `Attach local Skills to Console`. | Functional; click creates a `skills-context` `ChatHandoffPayload` from actual listed skill data rather than placeholder copy. | none |
| Library | Open Library with notes, media, and conversation services; source counts and sample titles render under `Local Library snapshot`. | Functional; `Use in Console` stages a `library-source-snapshot` payload and Import/Export plus Search/RAG remain reachable through explicit Library buttons. | none |
| Personas | Open Personas with local character/persona services; character and profile counts plus samples render under `Local Personas snapshot`. | Functional; `Attach to Console` stages a `personas-context` payload from actual behavior summaries. | none |
| W+C | Open W+C with watchlist and read-it-later services; monitored-source and saved-collection samples render under `Local W+C snapshot`. | Functional; `Stage W+C Context in Console` stages a `wc-context` payload while active-run Console follow remains intact. | none |
| Schedules | Open Schedules with active schedule-run adapter context or latest local reading-digest output. | Functional when actionable context exists; otherwise disabled Console recovery copy states that no active schedule run is available. | none |
| Workflows | Open Workflows with active workflow adapter context. | Functional when actionable context exists; otherwise disabled recovery copy states that no active workflow run is available. | none |
| Artifacts | Open Artifacts with local Chatbook records. | Functional when a Chatbook exists; otherwise disabled recovery copy or service-error copy explains why Console launch is unavailable. | none |
| Settings | Open Settings and click `Open Appearance`. | Functional; route posts `NavigateToScreen("customize")`, and Settings explicitly excludes MCP/tool control ownership. | none |
| ACP | Open ACP with no configured ACP-compatible runtime. | Verified honest recovery state; launch and Console follow buttons are disabled until runtime/session payloads exist. | none |

## Focused Verification

Commands were run from the repository root using the project virtual environment. Portable command form:

- Baseline replay: `python3 -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_shell_destinations.py Tests/UI/test_master_shell_navigation.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_unified_shell_phase234_maturity_gate.py -q`
- Baseline replay result before closeout tracker updates: `125 passed, 1 failed, 3 warnings`
- Baseline failure: `Tests/UI/test_unified_shell_phase234_maturity_gate.py::test_phase_two_three_four_closeout_tasks_record_current_parent_status` still expected `TASK-5.6` to remain `To Do` after the task was moved to `In Progress`.
- Red closeout contract: `python3 -m pytest Tests/UI/test_unified_shell_phase234_maturity_gate.py::test_phase_four_closeout_doc_records_verified_destinations_and_task_completion -q`
- Red closeout result before evidence existed: `1 failed` with `FileNotFoundError` for this closeout document.
- Final focused replay: `python3 -m pytest Tests/UI/test_unified_shell_phase234_maturity_gate.py -q`
- Final focused replay result after closeout updates: `5 passed`
- Final broader replay: `python3 -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_shell_destinations.py Tests/UI/test_master_shell_navigation.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_unified_shell_phase234_maturity_gate.py -q`
- Final broader replay result after closeout updates: `127 passed, 1 warning`

Warning boundary: the remaining warnings are existing dependency/import warnings and do not indicate destination workflow failures.

## UX Notes

- Destinations now describe ownership clearly enough for first-time users without reducing power-user density.
- Connected destinations expose concrete data or routes: MCP controls, Skills lists, Library source snapshots, Personas behavior snapshots, W+C monitored/saved source snapshots, Settings appearance routing, and Console staging where appropriate.
- Disabled controls are not ambiguous: ACP, empty Schedules, empty Workflows, and empty Artifacts states explain what is missing before the action can run.
- The tested flows are not render-only: mounted Textual tests click buttons and verify navigation messages, payload construction, app method calls, state restoration, service call parameters, unavailable states, and recovery copy.

## Defects And Blockers

No Phase 4 closeout blocker remains.

The only issue found during replay was stale tracker/test state from transitioning `TASK-5.6` into closeout. The destination wrapper, shell navigation, Console handoff, and maturity-gate checks passed after the closeout evidence and task status updates.

## Residual Risk

- Full Library-native detail views, embedded Import/Export, and embedded Search/RAG remain future work; current Library keeps those paths reachable through existing routes.
- Skills import, detail, edit, validation, execution, and full `SKILL.md` body staging remain future work.
- Personas detail/edit/import/export, archetypes, exemplars, dictionaries, lore, and full card staging remain future work.
- W+C create/edit/delete, import/export, WebSub, alert-rule editing, retry/backoff, and server collection-feed UX remain future work.
- ACP runtime/session management is not wired; Phase 4 verifies honest recovery state rather than an ACP session workflow.
- Full first-time-user and power-user shell replay remains Phase 6 scope.

## Closeout Decision: verified

Phase 4 is verified because each required destination now has at least one meaningful workflow or an explicit honest recovery state. The replay covers Skills, MCP, ACP, Library, Workflows, Schedules, Personas, Artifacts, W+C, and Settings ownership, and the evidence distinguishes implemented destination workflows from deferred service-depth work.
