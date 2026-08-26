---
id: TASK-16204
title: Preserve message trees during history resave
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 06:24'
updated_date: '2026-08-14 06:24'
labels:
  - chat
  - persistence
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make authoritative chat-history resaves update retained messages without triggering the interactive single-message edit rule that tombstones every descendant, while still deleting rows omitted from the resaved history.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Resaving a retained message tree updates the requested rows without tombstoning later retained turns or selected-variant groups.
- [x] #2 Rows omitted from an authoritative resave are still soft-deleted, and positional fallback ignores variant rows.
- [x] #3 Direct single-message content edits retain the existing descendant-tombstone behavior.
- [x] #4 Focused persistence, mutation, static, and diff gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this fixes an interaction between two existing persistence contracts without changing data ownership or schema.

1. Capture the failing explicit-id and positional resave cases plus the direct-edit descendant-tombstone characterization.
2. Add the narrowest persistence seam that lets authoritative bulk resaves preserve descendants while leaving ordinary message edits unchanged.
3. Prove mutations of the new seam fail the named resave tests, then run adjacent chat persistence and static/diff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added an explicit `preserve_descendants` bulk-update seam from `ChatPersistenceService.save_history` through the database updater. Authoritative resaves use it for both explicit-ID and positional matching, then retain their existing final soft-delete pass for rows omitted from history. Ordinary message updates keep the default descendant tombstone behavior. RED evidence: removing the explicit-ID flag failed the retained-tree/omitted-row test; removing the positional flag failed the variant-aware fallback test. GREEN: 157 adjacent chat persistence/history/tree tests passed, including the direct persisted-ancestor edit characterization. Scoped Ruff check and diff-check passed; Ruff format reports the same inherited whole-file drift and was not allowed to churn the two large production modules. ADR required: no; this reconciles existing edit and bulk-resave contracts without schema or ownership changes.
<!-- SECTION:NOTES:END -->
