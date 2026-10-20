---
id: TASK-35866
title: Keep later Results actions from cancelling an admitted review batch
status: To Do
created_date: 2026-10-20 19:24
labels:
- scheduling
- textual-patterns
- reliability
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-20-textual-book-reanalysis.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A Results review action must not silently discard an already admitted Mark all read batch. Overlapping review gestures need a clear outcome and truthful completion counts while preserving existing per-row review and outbox behavior.

### Evidence and bounded scope

results_tab.py:711-762 assigns Read, Dismiss, Mark solved and Mark all read to one exclusive schedules-results-host group. The batch awaits one service review per ID at :618-620. A private-profile real-keyboard/SQLite probe held the first get_automation_result thread, pressed a then d, and settled all workers: three initial unread rows ended unread, unread, dismissed, with the batch completion abandoned.

Book basis: Chapter 7 printed p.149 / PDF p.158; thread workers printed p.326 / PDF p.335. Latest-result cancellation must be adapted to admitted mutations rather than copied blindly from read-refresh examples.

### Existing ownership and decisions

The Study durable-write precedent TASK-19559 is Done. TASK-33283 owns Schedules loop blocking and redraw cost; TASK-34931 covered overlay preparation; TASK-33622.15 covered quit. This task covers overlapping hosted Results mutations only.

Related tasks: [TASK-19559](./task-19559%20-%20exclusive%3DTrue%20without%20group%3D%20makes%20every%20worker%20on%20a%20node%20cancel%20its%20siblings.md); [TASK-33283](./task-33283%20-%20PERF-24-Watchlists-and-Schedules---incremental-panes-DB-off-the-event-loop.md); [TASK-34931](./task-34931%20-%20Move-Schedules-results-overlay-and-mark-read-queries-off-the-UI-thread.md); [TASK-33622.15](./task-33622.15%20-%20CtrlQ-under-a-modal-in-flight-operations-and-the-video-save-picker-can-still-lose-work.md).

ADR required: no new ADR expected.
ADR paths: [018-local-server-hybrid-scheduled-tasks](../decisions/018-local-server-hybrid-scheduled-tasks.md); [116-schedules-inspector-editing](../decisions/116-schedules-inspector-editing.md).
Reason: No new ADR is expected if the repair stays within existing UI mutation admission and target ownership. Preserve Accepted ADR-018 storage/sync semantics; Proposed ADR-116 is contextual evidence, not ratified here.

[Post-merge audit and evidence](../../Docs/superpowers/qa/2026-10-20-textual-book-reanalysis.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With the first review read/write held, a later Read, Dismiss or Mark solved gesture cannot silently cancel an admitted Mark all read batch.
- [ ] #2 Overlapping gestures are explicitly refused while busy or safely serialized; accepted actions retain their intended result IDs and scope and publish truthful success/failure counts.
- [ ] #3 Per-row failures remain visible and retryable; ordinary single-row Read/Dismiss, Mark solved eligibility and global/per-definition bulk scopes keep their behavior.
- [ ] #4 Existing local/server owner checks and review-plus-outbox transaction semantics are preserved without a bulk schema primitive or new sync/conflict policy.
- [ ] #5 Deterministic actual-keyboard/private-profile SQLite tests cover a-then-d overlap, related action overlap, physical settlement and ordinary/failure controls.
<!-- AC:END -->
