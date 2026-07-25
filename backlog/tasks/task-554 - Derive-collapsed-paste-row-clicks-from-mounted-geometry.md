---
id: TASK-554
title: Derive collapsed-paste row clicks from mounted geometry
status: Done
assignee: []
created_date: '2026-07-25 17:08'
updated_date: '2026-07-25 17:12'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the Console collapsed-paste row-click test aligned with the mounted draft region instead of a layout-dependent hard-coded coordinate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The composer-row click targets the visible draft using mounted geometry
- [x] #2 Production composer behavior remains unchanged
- [x] #3 The focused row-click and adjacent paste interaction tests pass
- [x] #4 Task notes record RED evidence ADR decision and verification
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the isolated hard-coded-coordinate failure and verify direct draft activation still passes.
2. Replace the fixed row offset with an offset derived from the visible draft and composer regions.
3. Run the collapsed-paste interaction cluster and continue the full UI fail-fast slice.
4. Review and document the change.

ADR required: no
ADR path: N/A
Reason: This is a test-coordinate correction for an existing UI contract and changes no runtime behavior or application boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Summary: Replaced the Console collapsed-paste row-click test's hard-coded offset with an offset derived from the mounted visible-draft and composer regions.

RED evidence:
- The isolated row-click test consistently left the token at `Pasted Text: 160 Characters` because `(13, 1)` no longer landed on the visible draft after layout changes.
- Direct visible-draft clicks and `activate_visible_draft_screen_position(...)` tests continued to pass, confirming the production unfurl behavior was intact.

Verification:
- All collapsed-paste cases in the Console internals module: 15 passed.
- Full `Tests/UI/test_console_internals_decomposition.py`: 123 passed.
- Ruff check for the changed test file: passed.
- Commit diff check: passed.
- Review confirmed only the test coordinate changed; production composer code is unchanged.

ADR required: no
ADR path: N/A
Reason: This corrects a test coordinate for an existing mounted UI contract and changes no runtime behavior or application boundary.

Files modified:
- `Tests/UI/test_console_internals_decomposition.py`
- `backlog/tasks/task-554 - Derive-collapsed-paste-row-clicks-from-mounted-geometry.md`
<!-- SECTION:NOTES:END -->
