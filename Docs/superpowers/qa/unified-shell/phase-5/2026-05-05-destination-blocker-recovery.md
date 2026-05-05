# Phase 5.2 Destination Blocker Recovery

Date: 2026-05-05
Task: `TASK-6.2`
Parent Task: `TASK-6`
Branch: `codex/unified-shell-phase5-destination-recovery`
Base: `origin/dev` at `a9166379`

## Goal

Apply the Phase 5 shared recovery taxonomy to the first high-impact shell destination blocker family so disabled ACP, Schedules, Workflows, and Artifacts actions explain what is unavailable, why, what to do next, where recovery lives, who owns the blocker, and which stable selector tests the state.

## Applied Blockers

| Destination | Blocked workflow | Canonical state | Stable selector | Recovery behavior |
| --- | --- | --- | --- | --- |
| ACP | ACP agent launch | `runtime_not_configured` | `#acp-empty-state` | Configure an ACP runtime in Settings before launch. |
| Schedules | Console follow for Schedules | `empty_selection` | `#schedules-console-unavailable` | Start or select a schedule run before opening it in Console. |
| Workflows | Console launch for Workflows | `empty_selection` | `#workflows-console-unavailable` | Start or select a workflow run before opening it in Console. |
| Artifacts | Console launch for Chatbook artifacts | `empty_selection` | `#artifacts-console-unavailable` | Create or import a Chatbook artifact before opening it in Console. |

## UX Result

Each target disabled state now exposes:

- Status label, such as `Runtime not configured` or `Select an active run`.
- Unavailable workflow, such as `Console follow for Schedules`.
- Immediate reason, such as no active schedule run or no local Chatbook artifact.
- Next action in user language.
- Recovery target, such as Settings, Schedules, Workflows, or Artifacts.
- Authority owner, such as local app runtime or local service data.
- Stable selector and disabled tooltip copy.

## Verification

- Red regression: `python3 -m pytest Tests/UI/test_console_live_work_handoffs.py::test_phase_five_destination_blockers_expose_taxonomy_recovery_fields -q`
- Red result before implementation: `4 failed`
- Focused destination verification: `python3 -m pytest Tests/UI/test_destination_shells.py::test_automation_destination_wrappers_explain_ownership Tests/UI/test_console_live_work_handoffs.py::test_skeletal_destination_console_actions_are_disabled_with_recovery_copy Tests/UI/test_console_live_work_handoffs.py::test_schedules_destination_keeps_console_launch_disabled_without_digest_output Tests/UI/test_console_live_work_handoffs.py::test_schedules_destination_keeps_console_follow_disabled_without_active_run Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_keeps_console_launch_disabled_without_chatbooks Tests/UI/test_console_live_work_handoffs.py::test_phase_five_destination_blockers_expose_taxonomy_recovery_fields -q`
- Focused destination result: `13 passed`

## Residual Risks

- This slice covers the first shell destination blocker family only.
- Runtime-policy blockers such as wrong source, server auth, server session, and policy denial still need taxonomy application in later Phase 5 slices.
- Optional dependency blockers for local models, speech, transcription, and embeddings/RAG extras still need a separate pass.
- This QA evidence is focused widget/app harness verification, not the full Phase 6 first-time and power-user walkthrough.

## Closeout Decision: destination_blockers_applied

`TASK-6.2` applies the shared recovery taxonomy to the highest-impact shell destination blockers without marking Phase 5 complete. `TASK-6` remains open until the remaining blocker families are covered and running-app QA verifies recovery paths end to end.
