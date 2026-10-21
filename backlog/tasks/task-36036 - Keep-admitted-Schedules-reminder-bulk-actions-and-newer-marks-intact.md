---
id: TASK-36036
title: Keep admitted Schedules reminder bulk actions and newer marks intact
status: Done
created_date: 2026-10-21 16:12
labels:
- scheduling
- textual-patterns
- reliability
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-21-textual-book-reanalysis.md
assignee:
- '@codex'
updated_date: 2026-10-21 19:13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An accepted reminder bulk action must finish for its captured rows or report its actual partial outcome. Later bulk gestures must not silently abandon that work, and an older completion must not erase a newer selection.

### Evidence and scope

Verified on latest dev df24de7ad71b95992f63a77fc5a19726e840849d. schedules_workbench.py:5125-5155 and :5325-5359 await each delete/update inside exclusive same-verb worker groups, then clear the entire current mark set. Mounted private-profile keyboard/confirmation diagnostics used the real SchedulingService and file-backed SQLite with controlled server-client awaits. Two marked rows followed by a later action on a newly marked third called srv0 then srv2, never srv1, for both toggle and confirmed delete. Only the later one-row completion was reported. Separately, allowing the original toggle to finish erased the newer third-row mark. Local non-suspending writes were not implicated; cancelling an awaiter does not establish cancellation of a physical remote effect.

Book basis: Chapter 7 printed p.149 / PDF p.158 (asynchronous handlers), chapter 17 printed p.326 / PDF p.335 (grouped exclusive weather refresh and cancellation). The book supplies worker mechanics; protecting admitted mutations follows the measured defect and existing service contracts, not a blanket prohibition on cancellation.

### Existing ownership and decisions

TASK-35866 (Done) repairs Results review admission, not reminder queue batches. TASK-19559 (Done) fixes missing groups, while these groups are explicit. TASK-23107 (Done) covers mark visibility/fallthrough. TASK-33283 (To Do) and TASK-32804 (In Progress) own event-loop/performance work. None owns the demonstrated same-verb reminder-batch or newer-mark completion defect.

Related tasks: [TASK-35866](./task-35866%20-%20Keep-later-Results-actions-from-cancelling-an-admitted-review-batch.md); [TASK-19559](./task-19559%20-%20exclusive%3DTrue%20without%20group%3D%20makes%20every%20worker%20on%20a%20node%20cancel%20its%20siblings.md); [TASK-23107](./task-23107%20-%20Schedules-bulk-mark-mechanism-is-invisible-no-legend-count-or-bulk-mode-hint.md); [TASK-33283](./task-33283%20-%20PERF-24-Watchlists-and-Schedules---incremental-panes-DB-off-the-event-loop.md); [TASK-32804](./task-32804%20-%20Work-stream-synchronous-work-on-the-Textual-event-loop.md).

ADR required: no.
ADR paths: [018-local-server-hybrid-scheduled-tasks](../decisions/018-local-server-hybrid-scheduled-tasks.md); [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md); [116-schedules-inspector-editing](../decisions/116-schedules-inspector-editing.md).
Reason: A bounded UI operation/selection ownership correction retains SchedulingService, per-row owner checks, SQLite, server/offline outcomes and sync policy. ADR-018 and session-state ADR-033 are Accepted; ADR-116 is Proposed context only. New storage primitives, server atomicity/idempotency, cross-screen mutation lifetime and shutdown redesign require separate scope/ADR review.

[Latest-dev book audit and evidence](../../Docs/superpowers/qa/2026-10-21-textual-book-reanalysis.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 While a server-owned reminder batch is held at its first request, later bulk enable/disable or confirmed deletion cannot silently cancel or replace the admitted batch; overlap has clear refusal or safe serialization behavior.
- [x] #2 Every accepted batch retains its captured reminder IDs, owners and intended operation, and reports truthful per-row success/refusal/failure outcomes without claiming that an interrupted await cancelled a remote side effect.
- [x] #3 Completion cannot erase selection changes made after admission; newer marks remain available, or selection-changing gestures are clearly refused while the batch is pending.
- [x] #4 Deterministic mounted keyboard/confirmation tests with the existing private-profile and real SQLite services cover toggle/delete overlap, newer marks, ordinary completion and refusal/failure controls while preserving existing local/server and transfer guards.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR paths: backlog/decisions/018-local-server-hybrid-scheduled-tasks.md; backlog/decisions/033-application-session-state-ownership.md
Reason: bounded correction of UI admission/selection ownership; preserve SchedulingService, storage, per-row owner checks, server/offline/transfer and shutdown contracts. ADR-116 remains Proposed context.

1. Add deterministic mounted keyboard/confirmation regressions using the existing private-profile and real SQLite services. Prove same-verb toggle/delete overlap and newer marks fail before product edits; use controlled server-client awaits without claiming remote physical cancellation.
2. Admit one reminder bulk mutation synchronously, freeze intended IDs/owners/operation, clearly refuse overlapping mutations and selection changes while pending, and release admission on successful, refused, failed or scheduling-error settlement. Keep cursor navigation and existing service policies intact.
3. Cover ordinary and partial refusal/failure outcomes and post-settlement retry. Update scheduling architecture documentation; use targeted affected tests and scoped static checks only.
4. Obtain independent task review, then let the controller admit selected new regressions into the required UI census and run combined integration/CI checks. Complete task notes and criteria only after all evidence is available.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented one synchronous screen-local admission gate for reminder bulk toggle and confirmed deletion. Frozen IDs, owners and desired enabled values retain accepted intent; overlapping bulk/mark gestures are visibly refused. Finalization removes only captured marks and releases admission, including submission failure/retry; per-row service refusal/error counts remain truthful.

Changed schedules_workbench.py, new test_schedules_bulk_admission.py and scheduling-workflows.md. Shared CI integration appends representative regressions while retaining all 430 existing slots, thirteen shards and limits (434 entries total).

ADR required: no. Existing ADR-018/033 apply; ADR-116 remains Proposed context. No service, storage, owner, remote atomicity or cross-screen lifetime policy changed.

Evidence: all 14 distinct new regression nodes have passing receipts (12 unchanged passing nodes plus the two corrected notification-drain controls), four service controls pass, and independent Superpowers spec/quality review is Approved with no Critical/Important findings. Shared CI contracts: 121 pass; six new gated cases collect. Scoped lint/new-test formatting/UTF8/compile/whitespace checks pass. Eight existing plain UI controls stopped before their bodies in source-profile fixture admission; existing formatter/warning debt remains qualified rather than suppressed. Hosted results are tracked on the PR.

[Qualification and structured evidence](../../Docs/superpowers/qa/2026-10-21-textual-bulk-and-import-fixes.md).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
