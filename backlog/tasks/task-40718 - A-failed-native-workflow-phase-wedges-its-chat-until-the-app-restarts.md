---
id: TASK-40718
title: "A failed native workflow phase wedges its chat until the app restarts"
status: In Progress
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
- [ ] #2 Reconcile on a failed execution phase whose workers have all returned succeeds and records the review.
- [x] #3 A failed phase whose workers may still be live is still refused with `workflow_resources_still_live`.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. Reason: the failure path reuses the existing failed-phase settlement; no new contract or stored state.
1. Reproduce through the native workflow rig: a phase whose remaining task depends on a task it will never run raises "workflow dependencies blocked"; show the run stays `recovery_required`, the chat refuses a new turn, and reconcile refuses.
2. In `_execute`, catch an ordinary exception from the phase body. If the service still sees live workflow resources (`_has_live_workflow_resources`: waves, open preparations, admitted owners, live native threads), re-raise so custody is retained as today. Otherwise mark the phase failed and fall through to the existing retirement tail (cleanup accounting, close preparations, `settle_execution_phase(outcome="failed")`, publish, release). Settlement and close re-prove physical return; any refusal still retains custody.
3. Pin AC #3: a failure while a worker is live stays retained and reconcile refuses with `workflow_resources_still_live`.
4. AC #2 (reconcile after the retained phase's workers return) needs a decision about what reconcile records; probe and report.
5. Mutation checks, covering files base vs head, preflight, Tests/Architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
`ConsoleWorkflowCoordinator._execute` now catches an ordinary exception from the phase body. When `WorkflowRunService._has_live_workflow_resources` still sees work for the workflow (a wave, an open preparation, an admitted owner, a live native thread), it re-raises and the old handler keeps custody. Otherwise the phase is marked failed and runs the existing retirement tail: cleanup accounting, preparation close, `settle_execution_phase(outcome="failed")`, publish and `_release`. Those steps re-prove the physical return themselves, so a refusal anywhere in them still retains custody. A blocked phase now ends like a phase whose task failed: `paused` (stored `awaiting_resume`), resumable, and the chat gets its intake claim and primary token back. Cancellation is not caught.

Tests (`Tests/Chat/test_workflow_failed_phase_release.py`, native rig in a private child):
- `blocked_phase`: task-2 depends on task-1 and the driver's task read omits task-1. On base the run stayed `recovery_required` in `_runs`, a new turn was refused ("A turn is already preparing."), reconcile raised `workflow_resources_still_live`, and runtime disposal raised `workflow_original_shutdown_unresolved`. Now the run is released, the status is `paused` and a new turn is accepted. Making the handler always re-raise fails it again.
- `live_worker_failure`: the driver's wave wait raises while task-1's worker is held. The run stays retained, reconcile refuses with `workflow_resources_still_live`, and the failed-phase retirement is never attempted. Removing the live-resource check fails it, because the driver then attempts cleanup and close over live work (settlement still refuses, so custody held either way).

AC #2 is open: it needs a decision. After a retained phase's workers return, `_has_live_workflow_resources` is false, but reconcile refuses because the run has no `prepared`. The stored workflow was never put into recovery (`awaiting_approval`; `recovery_required` exists only in memory), so `resolve_recovery` answers `recovery_unavailable`, and the phase's native attempt is still `accepted`. Reconcile could either retire it as a failed phase (resumable, nothing reviewed) or record a durable execution recovery and close it as reviewed history (a new service step like `record_returned_planning`).

Scope note: any ordinary exception from the phase body with nothing live now settles as a failed phase, not only blocked dependencies. That includes admission refusals such as `AutomaticWorkRefused("wall_budget")` raised before launch, which on base wedged the chat as `recovery_required`. Failures in the retirement tail itself (cleanup accounting, close, settlement, publish, release) still retain custody as before.

Evidence (load 30-65 on 18 CPUs, so most native child tests time out on both trees): 13 covering workflow files, base 58 failed / 136 passed against head 71 / 123. Every head-only node was rerun alongside base at the same load. 16 of them fail on base too. `partial-resume` passes 2/2 on both when run alone, and `capture` plus the notification journey pass on head serially. `cleanup-expired` fails on both trees at the same assertion ("readiness wait never armed": the 10 s wall budget runs out before the wait is armed). Its full child logs show base then wedging (`recovery_required`, shutdown unresolved) while head settles the phase as `paused`. The two wave-probe nodes never enter the coordinator, and base raises their "workflow close wave changed" in sibling tests. The diagnostic inventory gained one constant-text `logger.exception("Native workflow phase failed")`; the pin was regenerated after reading it.

Files: `Chat/console_workflow_runs.py`, `Tests/Chat/test_workflow_failed_phase_release.py`, `Docs/security/production-diagnostic-inventory.json`, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
