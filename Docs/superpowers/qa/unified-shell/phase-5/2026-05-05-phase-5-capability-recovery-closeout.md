# Phase 5 Capability Recovery Closeout

Date: 2026-05-05
Task: `TASK-6.5`
Parent Task: `TASK-6`
Branch: `codex/unified-shell-phase5-closeout`
Base: `origin/dev` at `9d1cfe50`

<!-- PHASE_5_CLOSEOUT_METADATA:BEGIN -->
```json
{
  "closeout_task": "TASK-6.5",
  "parent_task": "TASK-6",
  "decision": "verified",
  "verified_recovery_families": [
    "destination-blockers",
    "runtime-policy",
    "optional-dependencies"
  ],
  "verified_blocked_states": [
    "runtime_not_configured",
    "empty_selection",
    "wrong_source",
    "server_auth_required",
    "server_session_invalid",
    "policy_denied",
    "dependency_missing"
  ],
  "required_recovery_fields": [
    "status_label",
    "unavailable_what",
    "why",
    "next_action",
    "recovery_action",
    "authority_owner",
    "stable_selector",
    "disabled_tooltip"
  ],
  "baseline_replay_result": {
    "passed": 93,
    "failed": 0,
    "deselected": 1,
    "warnings": 3
  },
  "red_closeout_contract_result": {
    "passed": 0,
    "failed": 1,
    "warnings": 1
  },
  "final_focused_replay_result": {
    "passed": 16,
    "failed": 0,
    "warnings": 1
  },
  "final_broader_replay_result": {
    "passed": 94,
    "failed": 0,
    "warnings": 1
  }
}
```
<!-- PHASE_5_CLOSEOUT_METADATA:END -->

## Current Baseline

Phase 5 is replayed after the merged capability and recovery slices:

- `TASK-6.1` defines the shared recovery taxonomy and required user-facing fields.
- `TASK-6.2` applies the taxonomy to ACP runtime, Schedules empty active-run, Workflows empty active-run, and Artifacts empty Chatbook blockers.
- `TASK-6.3` applies the taxonomy to runtime-policy denials in Skills, Library, Personas, and W+C.
- `TASK-6.4` applies the taxonomy to Search/RAG embeddings and local speech optional-dependency blockers.
- `TASK-6.5` records the maturity-gate replay and closeout decision.

The implemented boundary is blocked-state recovery, not full service completion. The shell now distinguishes unavailable capabilities from broken UI by showing what is blocked, why it is blocked, who owns the authority, what the user can do next, and where recovery should happen.

## Workflow Matrix

| Recovery Family | Running-app path | Result | Severity |
| --- | --- | --- | --- |
| Destination blockers | Open ACP, Schedules, Workflows, or Artifacts without actionable runtime/run/artifact context. | Disabled controls expose stable selectors, visible recovery copy, and tooltips instead of silent no-op actions. | none |
| Runtime-policy denials | Open Skills, Library, Personas, or W+C under wrong-source, auth-required, expired-session, or policy-denied service states. | Recovery copy maps reason codes to user-understandable next actions and authority owners. | none |
| Optional dependencies | Open Search/RAG without embeddings extras, or local speech without local TTS/STT extras. | Missing optional extras show install guidance, disabled tooltips, and persistent recovery copy without leaving the screen in an infinite loading state. | none |

## Focused Verification

Commands were run from the repository root using the project virtual environment. Portable command form:

- Red closeout contract: `python3 -m pytest Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py::test_phase_five_closeout_doc_records_verified_recovery_paths_and_task_completion -q`
- Red closeout result before evidence existed: `1 failed, 1 warning` with `FileNotFoundError` for this closeout document.
- Baseline recovery replay before closeout updates: `python3 -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_disabled_action_recovery_tooltips.py Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py -q -k "not closeout"`
- Baseline recovery replay result: `93 passed, 1 deselected, 3 warnings`
- Final focused closeout replay: `python3 -m pytest Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py -q`
- Final focused closeout replay result after closeout updates: `16 passed, 1 warning`
- Final broader recovery replay: `python3 -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_disabled_action_recovery_tooltips.py Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py -q`
- Final broader recovery replay result after closeout updates: `94 passed, 1 warning`

Warning boundary: the remaining warnings are existing dependency/import warnings and do not indicate recovery workflow failures.

## UX Notes

- Blocked states now use consistent language: status, unavailable item, reason, next action, recovery target, owner, selector, and disabled tooltip.
- Recovery copy is specific enough for first-time users without hiding power-user paths such as Console staging, direct destination routes, and existing legacy surfaces.
- Disabled controls are intentionally explanatory rather than merely inert, reducing the false-affordance risk that started the shell rescue work.
- The tested flows are not render-only: mounted Textual tests verify visible copy, disabled/enabled state, stable selectors, tooltip strings, event payloads, worker-safe loading behavior, and helper mappings.

## Defects And Blockers

No Phase 5 closeout blocker remains.

The only issue found during closeout was the expected missing closeout evidence, captured by the red regression. The recovery taxonomy, destination blockers, runtime-policy blockers, optional-dependency blockers, and closeout tracking passed after the closeout evidence and task updates landed.

## Residual Risk

- Auth and server recovery paths are verified through policy-reason mappings and shell states, not against a live remote server.
- Optional dependency recovery verifies missing dependency states and install guidance; it does not install or exercise every optional extra.
- ACP remains runtime-not-configured and recoverable, not a complete ACP session workflow.
- Full first-time-user, power-user, and Nielsen heuristic replay remains Phase 6 scope.

## Closeout Decision: verified

Phase 5 is verified because the shared recovery taxonomy is applied to representative destination blockers, runtime-policy denials, and optional-dependency blockers, and the running-app replay proves blocked states are understandable and recoverable rather than silent, generic, or visually broken.
