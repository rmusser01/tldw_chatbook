---
id: TASK-24205
title: Harden Folder path-task drafts and complete public API docs
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29 14:52'
updated_date: '2026-08-29 14:58'
labels:
  - library
  - notes
  - review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address Qodo review findings on PR #2200 by preventing New and Move path tasks from discarding post-flush edits and documenting the new public status and rewind APIs to repository standards.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 New and Move keep the exact current editor read-only from preflight flush through path-task completion or cancellation
- [x] #2 Save Copy retains its incumbent editable behavior and every path-task exit releases only its own read-only reason
- [x] #3 The rewind and both Notes status APIs have complete Google-style parameter and return documentation
- [x] #4 Focused path-task, status, rewind, lint, format, compile, and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: this uses the existing tokenized editor read-only ownership seam to fix a bounded draft-loss bug and adds compliance documentation; no storage, sync, service, or UX architecture changes. First add a regression that proves New/Move lock the editor through the path task and release on cancel while Save Copy remains editable. Then retain an exact path-task lease from before flush until close, add the three Google-style docstrings, run focused behavior/status/rewind tests and static checks, update the PR, and request Qodo re-review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented exact token-owned editor locking for New and Move from before the preflight flush until the path task closes, while preserving Save Copy editability and independent read-only reasons. Added focused regressions for the lock interval, cancellation, stale admission, independent lease ownership, and Save Copy behavior. Completed Google-style documentation for apply_rewind_position and both Notes status resolver APIs.

Verification: Folder path-task group 9 passed; Notes status group 14 passed; Console rewind UI 23 passed; Ruff check and format, compileall, and git diff --check passed. Full repository suite was intentionally not run per user instruction. ADR required: no; bounded fix using the existing editor-lease seam.
<!-- SECTION:NOTES:END -->
