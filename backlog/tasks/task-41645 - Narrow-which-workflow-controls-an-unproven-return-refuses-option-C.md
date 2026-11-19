---
id: TASK-41645
title: "Narrow which workflow controls an unproven return refuses (option C)"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- workflows
- console
priority: medium
references:
- tldw_chatbook/Chat/console_workflow_runs.py
- backlog/decisions
dependencies:
- TASK-41320
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Rewritten 2026-11-18 after the owner's decision "Real C, keep both": the original filing misdescribed the owner's 2026-11-17 choice. That choice was restart-only for a control or read whose outcome is truly unknown, plus a narrowing filed for later: an ordinary control refusal (Stop racing the end of a phase, refused by the phase with `workflow_authority_changed`) should count as a refusal, not as an unknown return, so it stops holding the chat until restart. The filing instead described narrowing which controls the unproven-read marker refuses. Both are kept, and the detail view is made usable under the marker so the controls that keep working are reachable.

**Why.** TASK-41320 made any Pause or Stop whose worker returned without a result an unknown control (`control_unknown`), which holds the chat until restart, even when the phase simply refused it before storing anything. It also left every workflow control failing, and the workflow detail view unreadable, while an unproven metadata read's marker is set, although most controls do not need that read.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Pause or Stop that the phase refuses before storing a decision, with its worker returned, storage returned and sources current, is reported as a refusal and does not hold the chat; a control whose outcome is unknown stays restart-only
- [x] #2 Under the unproven-read marker only controls whose completion needs a proven metadata read refuse; the others work
- [x] #3 Under the unproven-read marker the workflow detail view shows the workflow and its Stop, Pause, Resume, Accept and Remove controls work from it
- [x] #4 ADR-285 records which controls refuse and why, and the User Guide describes the behaviour
- [x] #5 Tests cover: Stop racing the end of a phase (chat accepts a new turn without restart), a refused and an allowed control under the marker, and Stop from the detail view under the marker
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes, a dated ADR-285 amendment (replaces parts of the TASK-41320 section).

1. Marker narrowing (done first, before the rewrite): under the unproven-read marker, refuse before any effect only the controls whose completion needs a proven metadata read (Recovery, Start, Approve, Revise); the others report "applied" (TASK-41647).
2. Refusal narrowing in `_native_control`: capture the phase's own `AutomaticWorkRefused`; treat it as a refusal only when it is the worker's whole outcome (exception identity), the read retired, sources are current and no decision with the control's id was stored (new service read). Red test first: Stop racing the end of a phase, held at its terminal publication. Mutation-check each guard.
3. Detail view: a coordinator `view` read that uses `get` normally and, under the marker, the service reads commands already use; the view names the marker and disables Approve and Revise. Red test first: the real detail view under the marker, its Stop button.
4. ADR-285 and User Guide; covering files on head, compared with base.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Part 1, marker narrowing** (`9d32341002`). The two unproven returns hold different resources. An unproven control return (`control_unknown`) is per run: only controls on that run refuse. The unproven-read marker is coordinator-wide and refuses every `_metadata` read. A control acts on that resource when its own completion needs a proven read: Recovery, and Start, Approve (of a phase) and Revise, whose new driver returns its chat only through a proven terminal publication. `_refuse_new_driver_if_unproven()` makes Start, Approve and Revise refuse with `workflow_return_unproven` before any effect. Stop, Pause, Resume, Accept and Remove keep working and report "applied" (TASK-41647).
- Test `test_unproven_read_refuses_approve_but_stop_still_applies`: under the marker, Approve is refused and changes nothing (no run, no primary claim, stored workflow equal, provider calls `["planning"]`); Stop then applies. Red with the guard taken out: Approve claimed the chat, started its driver and reported `workflow_command_applied`.

**Part 2, refusal narrowing** (`f7cec1efd9`). In `_native_control`, `controlled()` records an `AutomaticWorkRefused` raised by `phase._control` itself. For an execution control (it has a decision), that refusal is accepted when the producer's exception is that same object and `run_preparation_read` raised it (so neither `_remaining_storage_return` nor `worker_guard`'s exit replaced it), the read retired, `current()` holds before and after, and `WorkflowRunService.decision_recorded(decision.id)` (new, added to `call`'s allowed set) is False. The control then raises the refusal; `control_unknown` stays unset, `run.control` and `run.status` are unchanged. Anything else stays unknown. The phase sends its pause and stop signals only after its decision commits, so a refusal with no stored decision sent none. Planning Stop has no decision row and keeps the old rule.
- New `test_stop_racing_the_end_of_a_phase_is_a_refusal`: the driver is held at its terminal publication, after settlement retired the phase; Stop passes its revision check and gets `workflow_authority_changed`; `control_unknown` stays unset; the driver releases the chat and a new turn is accepted. Red on 32600a1c58 (shutdown then reported the held run); the full child log of the no-narrowing mutation below shows Stop raising "workflow control worker return unproven" and the driver then refusing release ("workflow original return unproven").
- New `test_a_refusal_after_a_stored_decision_stays_unknown`: `_control` stores the Stop decision, then raises; the control stays unknown ("workflow control refusal unproven") and reconcile refuses with `workflow_return_unproven`.
- `test_unproven_control_return_is_released_only_by_restart` (TASK-41320) changed: its plain patched refusal is now a refusal (asserted), and its unknown case is a Pause whose control read loses a storage handle (raised from a `finally`, as the real check does). A Pause is used because a repeated Stop at the same revision replays the recorded command.
- Mutation checks, each restored: no refusal narrowing makes the race test red (as above); dropping the stored-decision check makes the stored-decision test red (Stop raised `workflow_authority_changed`); dropping the exception-identity checks makes `unproven_control` red (the raw storage error came out instead of the unproven path).

**Part 3, detail view** (`f7cec1efd9`). `ConsoleWorkflowCoordinator.view` is the detail view's read: `get` normally; under the marker `service.get` plus `service.runtime_snapshot` (checked against each other) with the run's runtime status, the same service reads commands already make. `ConsoleWorkflowStatus` uses it for its refresh, its repair re-read and its pre-action read. Under the marker the view shows "Chatbook could not confirm that a Pause, Stop or background check finished cleanly. This view shows saved state. Restart Chatbook to approve or revise work and release held chats." and disables Approve and Revise. The compact workflow line and command read-backs still use the coordinator's read chain.
- New `test_detail_view_stops_a_running_phase_under_an_unproven_read`: a running phase held at its wave call; another chat's read sets the marker; the real `ConsoleWorkflowStatus` over the real coordinator renders the running workflow with Stop enabled and the marker notice; pressing Stop shows "Stop applied", the stop is stored, and the wave then launches nothing. Red on 32600a1c58 (Stop never enabled; wait timed out). Mutation: `view` always calling `get` makes it red the same way.
- The TASK-41320 detail-notice checks in `unproven_final_read` and `unproven_earlier_read` now expect the rendered view's marker notice instead of "Review unavailable".

**Follow-up 3 verified, not fixed.** A phase running when the marker is set fails at its next wave boundary: in a probe (not committed), task-1 completed, the mid-phase `_publish` was refused, the TASK-40718 handler failed the phase, task-2 never ran, the phase settled `failed` and the run stayed held until restart. Whether to keep running (spending work that cannot be released until restart) or stop at the boundary as now is a product choice; ADR-285 records it as a known limit.

**Evidence (2026-11-19, head vs base d0486978dd).**
- `Tests/Chat/test_workflow_failed_phase_release.py`: 14 passed at `-n 4` on head.
- Mutation checks, each restored: `view` always calling `get` -> `unproven_final_read`, `unproven_earlier_read`, `test_detail_view_stops_a_running_phase_under_an_unproven_read` red; refusal narrowing off (`refusal = None`) -> `test_unproven_control_return_is_released_only_by_restart`, `test_a_refusal_after_a_stored_decision_stays_unknown`, `test_stop_racing_the_end_of_a_phase_is_a_refusal` red.
- Test drift from the status screen reading `coordinator.view`: only `Tests/UI/test_workflow_setup_worker.py::test_refresh_error_survives_partial_dom_removal` (fake coordinator exposed `get`, now `view`). The other UI fakes/real-coordinator tests (`Tests/UI/test_workflow_*`, `test_console_workflow_command.py`) pass.
- Wider run at `-n 4`: Tests/Chat/test_workflow_*.py, test_console_workflow_dispatch.py, Tests/UI/test_workflow_*.py, test_console_workflow_command.py: 218 passed, 24 failed; all 24 fail identically on base d0486978dd (native wave/return/journey tests, under load). `Tests/Chat/test_console_workflow_scheduling.py`: 26 failed / 4 passed on head and on base (same counts). No failure is new on head.
- `Tests/Architecture/test_module_size_ratchet.py` + `test_screen_size_ratchet.py`: 234 passed (none of the three touched source files is pinned). `./scripts/preflight.sh`: all checks passed.


**Review round (2026-11-19).**
- `view` now refuses on `_closing` (top of the call and again after its reads), applies the `_metadata_floor` monotonic check `get` applies (a snapshot below the floor raises `workflow_snapshot_superseded`), and re-runs the source-identity checks that are valid without a read cell (`console_runtime`, `agent_bridge`, store, `service.db is bridge.runs_db`) before and after the reads. Test `test_detail_view_under_the_marker_refuses_a_closing_or_stale_read`: red on the previous tip (view served a below-floor snapshot and a closing coordinator), green now.
- The driver failure handler wraps each never-launched `close()` in try/except Exception, logs the item and error, and falls through to `_has_live_workflow_resources`, so the original failure stays the reported one and a preparation that is really still open keeps custody. Test `test_a_failing_close_in_the_failure_handler_keeps_the_original_failure` (a close that closes, then raises): red before (the raise replaced the original failure and the phase took the did-not-retire path), green after. A close that raises while leaving the preparation open still holds custody through the live check, as intended.
- `decision_recorded` reads `service.db`; the phase ledger is built by `db.automatic_work` (a new `AutomaticWorkLedger(self)` per call) but a phase can hold a ledger from elsewhere, and nothing proved `phase._ledger._db is service.db` in `_native_control`. Added that check on the refusal branch: if they differ the control stays unknown ("workflow control refusal unproven"). Not separately testable without forging a phase with a foreign ledger; guarded, not covered.
- `unproven_final_read` flake: passed 3 of 3 alone on base d0486978dd and on head at `-n 4` (27 passed: the release and setup-worker files); not reproduced, so pre-existing-vs-new is not established. It was not touched by this task's code paths beyond the detail-notice assertion.
- Re-run: Tests/Chat/test_workflow_failed_phase_release.py + Tests/UI/test_workflow_setup_worker.py at -n 4: 27 passed; size ratchets 234 passed; preflight passed.

Files: `tldw_chatbook/Chat/console_workflow_runs.py`, `tldw_chatbook/Agents/workflow_run_service.py`, `tldw_chatbook/Widgets/Console/console_workflow_status.py`, `Tests/Chat/test_workflow_failed_phase_release.py`, `backlog/decisions/285-native-phased-agent-workflows.md`, `Docs/User_Guide/console/agent-runs-and-tools.md`.
<!-- SECTION:NOTES:END -->
