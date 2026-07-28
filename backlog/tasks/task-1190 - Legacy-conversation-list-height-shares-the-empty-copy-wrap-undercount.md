---
id: TASK-1190
title: 'Legacy conversation-list height shares the empty-copy wrap undercount'
status: To Do
assignee: []
created_date: '2026-07-27 21:30'
labels: [console, ui, layout]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-1142 fixed the grouped browser's tray-height undercount (empty_copy Statics wrap to 2 lines at ~100-190-column widths, clipping later headers out of the hit-testable bounds). The transitional legacy path `_legacy_conversation_list_height` (taken when state.conversation_browser is None) still uses the flat `_CONVERSATION_BROWSER_EMPTY_COPY_HEIGHT = 1` constant — the same undercount class can clip the "New conversation" button composed after it. Reviewer-confirmed real; different code path and symptom from 1142 so filed separately.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Legacy-path empty-copy heights account for wrapping (reuse 1142's estimator) or the legacy path is retired.
- [ ] #2 A width-parameterized test pins the fix (or the retirement removes the surface).
<!-- AC:END -->
