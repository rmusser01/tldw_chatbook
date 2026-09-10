---
id: TASK-32237
title: >-
  Library Media reader: the More strip clips 'Move to trash' to 'Move to' at the
  3-column layout
status: In Progress
assignee: []
created_date: '2026-09-10 15:15'
updated_date: '2026-09-10 16:55'
labels:
  - library
  - media
  - layout
  - regression
  - critique-9
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #7 recorded the clip as a minor; the critique-8 test-health pass (task-32199, PR #2567) traced the red pin in Tests/UI/test_library_media_render_fixes.py and verified the clip is a real regression independent of the task-31979 Items-pane widening: the Reader More row's last action is cut to 'Move to' at the 3-column layout at 235x52. Eight names in that file stay red on dev; the More-row clip is one of three distinct causes named in task-32199's notes (the other two: _row_is_painted_focused in the frozen test_library_shell.py, and a custom items_width that no longer reaches the pane). Evidence: critique #9 register + task-32199 Implementation Notes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The More strip paints every action label in full ('Move to trash') at 235x52, 100x30 and 60x24, wrapping to a second row when needed
- [ ] #2 The red media_render_fixes pins that cover the More row are green with no assertion weakened
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run the two red More pins, record the clip.
2. Widen the More ItemGrid columns 16 -> 17 so the danger action's own 2-cell margin fits inside its column.
3. Add 60x24 coverage in Tests/UI/test_library_crit9_media_reader.py.
4. Live-verify at 235x52 / 100x30 / 60x24.
<!-- SECTION:PLAN:END -->
