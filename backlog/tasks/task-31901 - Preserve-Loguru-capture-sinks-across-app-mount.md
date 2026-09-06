---
id: TASK-31901
title: Preserve Loguru capture sinks across app mount
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 09:26'
updated_date: '2026-09-05 09:29'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep worker-warning tests attached to the Loguru lifecycle they observe so app logging initialization cannot invalidate their owned sink.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Worker-warning capture tests install their sink after app logging initialization.
- [x] #2 Known navigation workers remain silent while unknown worker groups still warn.
- [x] #3 The complete `Tests/App` suite passes.
- [x] #4 Scoped Ruff and diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce both worker-warning capture failures independently.
2. Install each test-owned Loguru sink only after the mounted app finishes logging setup.
3. Run the focused worker event file and complete App suite.
4. Run scoped static and diff checks.

ADR required: no

ADR path: N/A

Reason: this repairs test resource ownership without changing logging architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved both warning-capture sinks inside the mounted-app context, after `_setup_logging` has removed and replaced process-wide Loguru handlers. Each test now owns its sink for exactly the observation window and removes it before app teardown. Verification: the focused file passed 4 tests, `Tests/App` passed 254 tests, Ruff passed, and `git diff --check` passed.
<!-- SECTION:NOTES:END -->

## PR 2427 rebase renumbering provenance

Review-owned TASK-31714 was renumbered to TASK-31901 on 2026-09-06
while rebasing PR 2427 onto dev c4d45c0926. The user approved preserving
upstream task identities and renumbering review-created collisions only.
Original creation dates, task history, and literal verification artifact paths
are retained. See backlog/docs/pr-2427-rebase-reconciliation.md for the mapping.
