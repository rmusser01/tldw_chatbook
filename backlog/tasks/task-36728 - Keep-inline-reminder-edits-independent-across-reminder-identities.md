---
id: TASK-36728
title: Keep inline reminder edits independent across reminder identities
status: In Progress
created_date: 2026-10-25 03:46
labels:
- scheduling
- textual-patterns
- correctness
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-24-textual-book-after-pr3243.md
assignee:
- '@codex'
updated_date: 2026-10-25 17:44
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An accepted inline edit must retain its reminder target while the user edits another reminder. A later edit on a different reminder must not silently abandon the earlier pending save.

### Evidence and bounded scope

Verified after PR #3243 merged, on dev `139333920d2a87cb011e4f3aa5f8319b59c44a5c`. On the audited dev commit, the real SchedulesWorkbench and SchedulingService were mounted over private SQLite. Holding server reminder A's timezone request, selecting B and changing B's timezone cancels A's worker: A remains UTC while B becomes Europe/London, with no warning. Serial control saves A=America/New_York and B=Europe/London. A same-reminder control retains the later Europe/London intent. The client is a controlled fake; remote physical cancellation, natural incidence and native dropdown browsing are not claimed. Only timezone was dynamically witnessed; the source shares the grouping mechanism with other inline fields.

Affected source: tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py:3084-3142; UI/Screens/scheduling/task_detail.py:1349-1360; Scheduling/services/scheduling_service.py:632-741.

Book basis: Chapter 7 PDF145-146/158 (printed136-137/149) describes queues and scheduling boundaries; chapter17 PDF335 (printed326) demonstrates grouped, cancellation-aware workers. Independent durable reminder edits need application-specific ownership; latest-result cancellation from a read example must not be copied across unrelated mutations.

### Existing ownership and decisions

TASK-19559 adds named worker groups; TASK-19870 serializes list refreshes; TASK-31711/31712 cover timezone and inspector affordances; TASK-33283 owns loop I/O cost; TASK-35866 owns hosted Results actions; TASK-36036 owns marked-reminder bulk actions. Their complete criteria do not own cross-reminder inline save cancellation.

Related tasks: [TASK-19559](./task-19559%20-%20exclusive%3DTrue%20without%20group%3D%20makes%20every%20worker%20on%20a%20node%20cancel%20its%20siblings.md), [TASK-19870](./task-19870%20-%20Mutation-workers-refresh-inline-instead-of-dispatching-into-the-loader-group.md), [TASK-31711](./task-31711%20-%20Schedules-timestamp-timezone-display-pass.md), [TASK-31712](./task-31712%20-%20Schedules-form-detail-pane-polish.md), [TASK-33283](./task-33283%20-%20PERF-24-Watchlists-and-Schedules---incremental-panes-DB-off-the-event-loop.md), [TASK-35866](./task-35866%20-%20Keep-later-Results-actions-from-cancelling-an-admitted-review-batch.md), [TASK-36036](./task-36036%20-%20Keep-admitted-Schedules-reminder-bulk-actions-and-newer-marks-intact.md).

ADR required: no.
ADR paths: [018-local-server-hybrid-scheduled-tasks](../decisions/018-local-server-hybrid-scheduled-tasks.md), [116-schedules-inspector-editing](../decisions/116-schedules-inspector-editing.md).
Reason: Bounded correction within the existing inline-edit owner and mutation service. Preserve Accepted ADR-018 ownership/sync policy; ADR-116 remains Proposed context, not ratified by this repair. Re-evaluate if durable mutation, transfer or conflict policy changes.

[Post-merge audit and evidence](../../Docs/superpowers/qa/2026-10-24-textual-book-after-pr3243.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 While an inline timezone save for reminder A is pending, an accepted timezone edit for distinct reminder B cannot silently cancel A; each accepted edit retains its original reminder ID and submitted value, or a truthful explicit refusal is shown.
- [ ] #2 Ordinary serial edits still save both reminders, and a later intent for the same reminder and field cannot be overwritten by an older completion.
- [ ] #3 Existing local/server validation, failure and retry, queued/offline and transfer restrictions remain intact; cancellation is never presented as proof of remote rollback.
- [ ] #4 Deterministic mounted tests exercise the real reminder detail events, service and private SQLite with a held client await, cover cross-reminder overlap, same-reminder and serial controls, and observe settlement and user-visible failure outcomes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the task, existing scheduling owner, lessons and ADR context; reproduce pending cross-reminder edits through real mounted rows and service/private SQLite. 2. Add regressions for independent and same-reminder targets plus stale row feedback, serial success and failure/refusal. Observe expected RED before production changes. 3. Bind existing worker exclusivity to reminder identity plus field, and restrict row feedback to its captured target while preserving service/refresh semantics. 4. Run targeted new and affected existing tests and scoped lint/source guards; request fresh Superpowers branch review. 5. Record evidence, check all ACs and mark Done through CLI after verification. ADR required: no. ADR path: backlog/decisions/018-local-server-hybrid-scheduled-tasks.md; backlog/decisions/116-schedules-inspector-editing.md (Proposed context). Reason: bounded UI owner/publication correction, no service/storage/sync policy change. Full plan: Docs/superpowers/plans/2026-10-25-textual-reminder-and-quiz-followups.md.
<!-- SECTION:PLAN:END -->
