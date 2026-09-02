---
id: TASK-26041
title: Review deferred hermes parity items after the 25900/26000-series burn-down
status: To Do
assignee: []
created_date: '2026-08-31 15:56'
labels:
  - parity
  - audit
  - review
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Holding task for the 73 capability rows from the 2026-08-31 hermes parity report that were reviewed and deliberately not filed. The list, with per-row reasoning, is backlog/docs/hermes-parity-deferred-items.md; the evidence behind each row is qa/hermes-parity-2026-08-31/report.md and gap-candidates.md. Owner's direction was to revisit these only after the 56 filed tasks (TASK-25900-25914, TASK-26001-26040, and TASK-28227, formerly TASK-26000) are burned down. This exists so the deferred set is recoverable from the board rather than surviving only in a report nobody re-reads. It is a review task: its output is decisions, not code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The deferred list is re-checked against a freshly fetched hermes origin/main - rows describing behavior that has since changed are corrected or dropped
- [ ] #2 The deferred list is re-checked against chatbook origin/dev - rows closed as a side effect of the filed 56 are marked closed with the evidence, not re-filed
- [ ] #3 Each remaining row gets a recorded decision: file it, fold it into an existing task, or reject it with a reason
- [ ] #4 Rows listed as blocked behind a filed task are resolved inside that task's outcome rather than becoming separate tasks
- [ ] #5 Any tasks created carry IDs swept against all remote refs and worktrees at creation time, per lessons-backlog-hygiene
- [ ] #6 The rejection table in the deferred-items doc is updated so a future reader can see what was declined and why
<!-- AC:END -->
