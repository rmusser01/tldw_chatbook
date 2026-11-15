---
id: TASK-40718
title: "A failed native workflow phase wedges its chat until the app restarts"
status: Done
created_date: 2026-11-13 06:28
labels:
- follow-up
- agents
- workflows
- bug
priority: medium
references:
- tldw_chatbook/Chat/console_workflow_runs.py
- tldw_chatbook/Chat/console_workflow_preparation.py
- https://github.com/rmusser01/tldw_chatbook/pull/3480
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-11-10 review of PR #3480 (TASK-39147, native phased workflows), which merged on 2026-11-12. It is still present on dev 7579f75ea9, re-checked on 2026-11-13 by reading the code. Line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** This applies only with `workflow_runs_enabled` on. A phase's `_execute` can raise, including on logical failures such as `ValueError("workflow dependencies blocked")` (`Chat/console_workflow_runs.py:1994`). Once it does, the chat refuses new turns until the app restarts.

**Cause.**
- The handler at :2111-2115 only records final metadata, which sets `status="recovery_required"` (:1540-1541).
- `_release` (:2142) runs only on success (:2110). It returns the intake claim, the automatic-primary token and the `_runs` entry.
- `_release_admission` refuses while the status is `recovery_required` (:1684-1685).
- Reconcile cannot clear it either. Execution runs are built without `prepared` (:1585-1595); only planning preparation sets it (`Chat/console_workflow_preparation.py:3103`). So `reconcile` raises `workflow_resources_still_live` (:2399-2400).

The handler's comment says a physical failure cannot shed custody. A failure that leaves no physical work running is caught by the same handler.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A phase that fails without leaving workers running, for example on blocked dependencies, releases its claim, and the chat accepts a new turn without a restart.
- [x] #2 Reconcile on a failed execution phase whose workers have all returned succeeds and records the review.
- [x] #3 A failed phase whose workers may still be live is still refused with `workflow_resources_still_live`.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. Reason: the failure path reuses the existing failed-phase settlement; no new contract or stored state.
1. Reproduce through the native workflow rig: a phase whose remaining task depends on a task it will never run raises "workflow dependencies blocked"; show the run stays `recovery_required`, the chat refuses a new turn, and reconcile refuses.
2. In `_execute`, catch an ordinary exception from the phase body. If the service still sees live workflow resources (`_has_live_workflow_resources`: waves, open preparations, admitted owners, live native threads), re-raise so custody is retained as today. Otherwise mark the phase failed and fall through to the existing retirement tail (cleanup accounting, close preparations, `settle_execution_phase(outcome="failed")`, publish, release). Settlement and close re-prove physical return; any refusal still retains custody.
3. Pin AC #3: a failure while a worker is live stays retained and reconcile refuses with `workflow_resources_still_live`.
4. AC #2 (owner decision 2026-11-13: close it as reviewed history). Add `record_returned_execution` to the workflow service, modelled on `record_returned_planning`. It refuses while `_has_live_workflow_resources` reports anything, retires the phase's authority, moves its native attempt to review, closes the clock interval and puts the workflow in `recovery_required`. Reconcile on an execution run whose driver is done and whose control return is proven calls it, then `resolve_recovery`, then returns the chat's intake claim and primary token. Reproduce first: the live-worker scenario extended past the worker's return.
5. Mutation checks, covering files base vs head, preflight, Tests/Architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
`ConsoleWorkflowCoordinator._execute` now catches an ordinary exception from the phase body. When `WorkflowRunService._has_live_workflow_resources` still sees work for the workflow (a wave, an open preparation, an admitted owner, a live native thread), it re-raises and the old handler keeps custody. Otherwise the phase is marked failed and runs the existing retirement tail: cleanup accounting, preparation close, `settle_execution_phase(outcome="failed")`, publish and `_release`. Those steps re-prove the physical return themselves, so a refusal anywhere in them still retains custody. A blocked phase now ends like a phase whose task failed: `paused` (stored `awaiting_resume`), resumable, and the chat gets its intake claim and primary token back. Cancellation is not caught.

Tests (`Tests/Chat/test_workflow_failed_phase_release.py`, native rig in a private child):
- `blocked_phase`: task-2 depends on task-1 and the driver's task read omits task-1. On base the run stayed `recovery_required` in `_runs`, a new turn was refused ("A turn is already preparing."), reconcile raised `workflow_resources_still_live`, and runtime disposal raised `workflow_original_shutdown_unresolved`. Now the run is released, the status is `paused` and a new turn is accepted. Making the handler always re-raise fails it again.
- `live_worker_failure`: the driver's wave wait raises while task-1's worker is held. The run stays retained, reconcile refuses with `workflow_resources_still_live` and changes nothing stored, and the failed-phase retirement is never attempted. Once the worker returns, the same reconcile closes the run (stored `closed`, `recovery_reviewed_revision` set; the coordinator displays closed history as `stopped`), the run leaves the coordinator, a new turn is accepted and shutdown is clean. Mutation checks: removing the driver's live-resource check fails it (cleanup and close are attempted over live work, though settlement still refuses); removing the record step's live check fails "the refusal recorded nothing"; removing reconcile's return of the chat claim fails the new turn ("A turn is already preparing.").

AC #2 (owner decision 2026-11-13: close it as reviewed history). `WorkflowRunService.record_returned_execution` is the execution counterpart of `record_returned_planning`. It refuses with `workflow_resources_still_live` while `_has_live_workflow_resources` reports anything. Otherwise it retires the phase's authority and, under the runtime owner, closes the clock interval (an unknown or reversed clock is kept, never invented). It also moves the native attempt to `review_required` and the phase attempt to `uncertain`/`unknown`, sets the workflow to `recovery_required`, and re-installs the folder protection. A retry after a recorded return changes nothing. For a run with no planning preparation, `reconcile` now requires a finished driver, a phase and a proven control return (no `control_unknown`, no unset `control_return`). It then calls the step and `resolve_recovery`, which closes the workflow and records the review, and returns the chat's intake claim and primary token before dropping the run. Reproduced first: on the AC #1 head the returned-worker reconcile still raised `workflow_resources_still_live` and shutdown stayed unresolved.

AC #2 evidence (load 43-68): six covering files (workflow service, recovery, recovery cancellation, coordinator ownership, native planning, history controls) were run on both trees. Base: 31 failed / 70 passed. Head: 24 failed / 79 passed. No node failed on head only. The shared failures are the load-sensitive ones seen in the AC #1 round (fleet threads still alive, child timeouts).

Review fixes (independent review, no Critical or Important findings):
- **M1.** `record_returned_execution` now runs its transaction and the phase retirement under `phase._control_lock`, and retires only after the record commits. A record refused inside the transaction (`revision_conflict`, `workflow_execution_return_required`) leaves the phase holding its authority and retryable. Pause, stop and launch commits take that lock before the ledger, so none can run between the record and the retirement. The live check stays just before the lock rather than inside it. The launch path (`agent_native_execution.py`) takes a task authorization's `_lock` and then `phase._control_lock`, and `_has_live_workflow_resources` takes those same task locks, so checking under the control lock would invert that order and could deadlock against an in-flight launch. Once the driver is done and nothing is live, nothing can start new work, and a racing pause or stop changes the revision, so the record fails retryably. Test: a reconcile with a stale revision raises `revision_conflict` and leaves the phase unretired and the store unchanged; moving the retirement back before the transaction fails it.
- **M2.** Reconcile returns the chat's claim through `_return_reviewed_admission`. Each claim (intake under `store._preparation_lock`, primary token under `wake._registry_lock`) is released only while the run still holds it, and a release that then reports `False` raises. A reconcile after a raised release finishes it. Test: the token release raises once, the run stays, and the next reconcile closes it and the chat accepts a new turn. The reviewer predicted the retry would fail inside `record_returned_execution`; it did not. The pre-review release code also passes this test, because both release helpers free only the exact claim and return `False` otherwise, and the step accepts a closed, reviewed run without change. The checks matter only for a race the test cannot stage.
- **Found while pinning M3.** A run kept after its final metadata was published leaves that metadata cell registered in `_metadata_cells` as a retention marker, and app shutdown refuses (`workflow_original_metadata_shutdown_unresolved`) while any marker remains. Reconcile left it behind. It now drops the run's marker when it returns the claim, and refuses beforehand (`workflow_resources_still_live`) unless the cell's read finished with a proven `True` disposition. A read whose return is unproven is never dropped. Leaving the marker in place fails the M3 test at shutdown.
- **M3 pin.** `ownership_error`: the phase body raises an ownership `PermissionError` after the session's primary token passed to another owner. The failed-phase path runs, but `_release`'s identity check refuses, so the other owner keeps the token and the run stays retained, refusing new turns. Once ownership is restored, reconcile closes the run (its native attempt was already settled), returns the claim, and the chat accepts a new turn.

Review-round evidence (load 13-20): the same six covering files, base 8 failed / 93 passed against head 8 failed / 95 passed, with the identical 8 nodes on both trees. All 8 are native planning: `test_native_planning_refuses_replaced_receivers_without_effects` (4), `test_native_planning_required_failure_retains_original_custody` (3) and `test_shutdown_waits_for_same_native_planning_worker_return`. At this low load they look dev-red rather than load-sensitive. Preflight passes.

Limit: a retained run that still holds an open, never-launched preparation counts as live, so reconcile keeps refusing it until restart, as before. The step does not close such preparations.

Scope note: any ordinary exception from the phase body with nothing live now settles as a failed phase, not only blocked dependencies. That includes admission refusals such as `AutomaticWorkRefused("wall_budget")` raised before launch, which on base wedged the chat as `recovery_required`. Failures in the retirement tail itself (cleanup accounting, close, settlement, publish, release) still retain custody as before.

Tests/Architecture (serial, final head): 3 failed / 1348 passed. All 3 fail identically on base 1455ceeb96: the `agent_service.py` diagnostic inventory and two `ProjectIndexBindingCard` timer-path checks. Preflight passes.

Evidence (load 30-65 on 18 CPUs, so most native child tests time out on both trees): 13 covering workflow files, base 58 failed / 136 passed against head 71 / 123. Every head-only node was rerun alongside base at the same load. 16 of them fail on base too. `partial-resume` passes 2/2 on both when run alone, and `capture` plus the notification journey pass on head serially. `cleanup-expired` fails on both trees at the same assertion ("readiness wait never armed": the 10 s wall budget runs out before the wait is armed). Its full child logs show base then wedging (`recovery_required`, shutdown unresolved) while head settles the phase as `paused`. The two wave-probe nodes never enter the coordinator, and base raises their "workflow close wave changed" in sibling tests. The diagnostic inventory gained one constant-text `logger.exception("Native workflow phase failed")`; the pin was regenerated after reading it.

Files: `Chat/console_workflow_runs.py`, `Agents/workflow_run_service.py`, `Tests/Chat/test_workflow_failed_phase_release.py`, `Docs/security/production-diagnostic-inventory.json`, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
