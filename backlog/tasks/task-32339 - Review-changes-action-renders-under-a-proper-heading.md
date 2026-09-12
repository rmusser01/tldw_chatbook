---
id: TASK-32339
title: >-
  Review changes action renders under a proper heading
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review D3. The 'Changes' entry exists in _ACTION_GROUPS but has no matching group in _ROW_GROUPS, so the review-changes button falls into the ungrouped actions loop (console_run_inspector.py ~131-135, ~428-431). Add the heading or re-home the action so it renders grouped like its siblings.

Filed from the 2026-09-10 Console rail UX review (review item D3).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The review-changes action renders inside a labeled group consistent with the run inspector's grouping
- [x] #2 No other action's grouping changes
- [x] #3 Widget test asserts the button renders under the expected heading
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
**Close-out.** Closed without code change: re-verification against dev tip found the Changes group heading already present and pinned by tests.

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

ALREADY FIXED ON DEV: 'Changes' is a full ROW_GROUP with heading (console_inspector_ownership.py:141; console_run_inspector.py:638-669), pinned by Tests/UI/test_console_run_inspector.py:394,418. Closing.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
