---
id: TASK-2330
title: check-now-failure test asserts a method the screen no longer defines
status: Done
assignee:
  - '@codex'
created_date: '2026-08-04'
updated_date: '2026-08-29 02:30'
labels:
  - watchlists
  - tests
dependencies: []
priority: medium
---

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run the complete Watchlists check-now failure-policy module on current dev to establish whether the obsolete logical action case is already reconciled.
2. If the test is green, retain the existing _delete_item to _update_item_status owner mapping and document why it remains discriminating; if red, make only the minimal test-contract repair and rerun the failing case.
3. Run the complete focused module, Ruff lint and format checks for modified Python files, and git diff --check.
4. Complete the acceptance criteria and Implementation Notes, then mark TASK-2330 Done only after fresh verification.

ADR required: no
ADR path: N/A
Reason: test-contract maintenance for an existing Watchlists writer path; no storage, sync, ownership, service, security, dependency, or long-lived UX boundary changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Current `dev` already contains the minimal repair: the legacy logical
`_delete_item` audit node is retained in `USER_INITIATED_MUTATIONS`, while
`USER_INITIATED_MUTATION_OWNERS` resolves it to the live
`_update_item_status` writer before source inspection. This preserves the
delete-action regression identity without asserting that the removed method
still exists, and the same parametrized contract separately audits the live
writer directly. No production or test-code change was needed.

Verification: the formerly failing `_delete_item` parameter passed alone;
the complete `Tests/UI/test_watchlists_check_now_failure.py` module passed
27 tests with two dependency warnings; Ruff lint and format checks passed for
that module; and `git diff --check` passed. ADR required: no. ADR path: N/A.
Reason: this closes stale test-contract bookkeeping for an existing writer
path without changing storage, sync, ownership, service, security,
dependency, or long-lived UX boundaries.

Modified files: this task file only (Backlog.md normalized its filename while
moving the task into progress).
<!-- SECTION:NOTES:END -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_watchlists_check_now_failure.py` has a parametrized case
(`[_delete_item]`) that fails on dev: it asserts against a method the screen
no longer defines (`_delete_item` was removed when the `d` binding was routed
through the TASK-1541 drain in PR #1342). Found during UAT batch-2 review and
verified failing on `origin/dev` at the batch-2 merge base (`ab9105c9d`) —
pre-existing there, not introduced by any open branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [x] #1 The parametrized case either targets the real current writer path or is
      removed with the reasoning recorded.
- [x] #2 The whole file passes on dev.
- [x] #3 The exemption contract the file documents (debug-level loader logging
      requires a failure toast) still has a discriminating case per loader.
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->
<!-- AC:END -->
