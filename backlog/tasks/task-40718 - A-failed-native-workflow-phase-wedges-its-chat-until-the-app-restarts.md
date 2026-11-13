---
id: TASK-40718
title: "A failed native workflow phase wedges its chat until the app restarts"
status: To Do
created_date: 2026-11-13 06:45
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

**Why.** This applies only with `workflow_runs_enabled` on. A phase's `_execute` can raise, including on logical failures such as `ValueError("workflow dependencies blocked")` (`Chat/console_workflow_runs.py:1991`). Once it does, the chat refuses new turns until the app restarts.

**Cause.**
- The handler at :2111-2115 only records final metadata, which sets `status="recovery_required"` (:1540-1541).
- `_release` (:2141) runs only on success (:2110). It returns the intake claim, the automatic-primary token and the `_runs` entry.
- `_release_admission` refuses while the status is `recovery_required` (:1684-1685).
- Reconcile cannot clear it either. Execution runs are built without `prepared` (:1585-1595); only planning preparation sets it (`Chat/console_workflow_preparation.py:3103`). So `reconcile` raises `workflow_resources_still_live` (:2398-2399).

The handler's comment says a physical failure cannot shed custody. A failure that leaves no physical work running is caught by the same handler.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A phase that fails without leaving workers running, for example on blocked dependencies, releases its claim, and the chat accepts a new turn without a restart.
- [ ] #2 Reconcile on a failed execution phase whose workers have all returned succeeds and records the review.
- [ ] #3 A failed phase whose workers may still be live is still refused with `workflow_resources_still_live`.
<!-- AC:END -->
