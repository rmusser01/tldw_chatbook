---
id: TASK-16262
title: Stabilize Library Notes recompose assertions
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 03:59'
updated_date: '2026-08-14 04:11'
labels:
  - testing
  - library
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Library Notes interaction tests wait for the post-recompose list and rail widgets before asserting their contents, preserving the user-visible behavior under Textual's asynchronous mount lifecycle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Receipt-dismiss waits for the Notes rail replacement before checking its count
- [x] #2 Undo interlock waits for the post-delete Notes row before interacting
- [x] #3 The three reproduced Notes tests pass and the Notes slice shows no regression from these waits
- [x] #4 Ruff check and diff checks pass; the patch preserves the file's pre-existing formatter state
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the reproduced RED traces showing post-recompose rail/list widgets are queried before remount.
2. Reuse existing selector wait helpers at the two stale assertions without changing production behavior.
3. Re-run the three named tests, affected Library module, Ruff, and diff checks; mutation-restore each stale assertion to prove it fails.
4. Record implementation notes and completion evidence.

ADR required: no
ADR path: N/A
Reason: test synchronization only; no product architecture or lifecycle policy changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added two bounded `_wait_for_selector` calls at the exact post-recompose gaps: the replacement Notes rail row after receipt dismissal and the remaining note row while Undo owns the mutation lock.
- Preserved the original RED (two failures/one pass), then verified the three named tests pass. The 260-test Notes slice produced 258 passes; its two independently reproducible focus/worker-baseline failures are recorded separately as TASK-16075 and TASK-16076.
- No production files changed. Ruff check and `git diff --check` pass. Ruff format still reports the test file's pre-existing whole-file formatting drift; an attempted mechanical format was fully reversed so this patch adds only the two synchronization lines.
- ADR required: no; this is test synchronization only.
<!-- SECTION:NOTES:END -->
