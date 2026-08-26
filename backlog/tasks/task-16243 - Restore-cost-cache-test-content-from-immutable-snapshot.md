---
id: TASK-16243
title: Restore cost-cache test content from immutable snapshot
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 10:26'
updated_date: '2026-08-14 10:36'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the cost estimate cache regressions isolated to a single editable leaf row and restore its original text from an immutable snapshot.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both one-row cache tests edit the conversation leaf without purging unrelated descendants.
- [x] #2 The restore test snapshots original content before editing.
- [x] #3 The restored cost tooltip matches the original.
- [x] #4 The full cost-cache module and affected sweep chunk are green.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm `update_message_content` treats an edited non-leaf as a branch rewrite and purges its descendants.
2. Target the leaf message in both single-row cache tests; capture the original string before the edit and restore from that immutable value.
3. Run the focused regression, full cost-cache module, sweep chunk, and static checks.

ADR required: no
ADR path: N/A
Reason: This is a test-fixture correction for existing conversation-tree and live-object store contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Moved both single-row estimate-cache edits to the active conversation leaf so the store does not correctly invalidate descendant history during a cache-only test.
- Captured the leaf's original string before mutation and restored from that immutable value.
- Verified the two focused edit/restore cases, all five estimate-cache tests, and the exact sweep chunk (542 passed). Ruff check/format and diff hygiene pass.
<!-- SECTION:NOTES:END -->
