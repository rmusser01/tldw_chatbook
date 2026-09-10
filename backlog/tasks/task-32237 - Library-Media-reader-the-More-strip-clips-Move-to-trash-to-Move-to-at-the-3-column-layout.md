---
id: TASK-32237
title: >-
  Library Media reader: the More strip clips 'Move to trash' to 'Move to' at the
  3-column layout
status: Done
assignee: []
created_date: '2026-09-10 15:15'
updated_date: '2026-09-10 17:40'
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
- [x] #1 The More strip paints every action label in full ('Move to trash') at 235x52, 100x30 and 60x24, wrapping to a second row when needed
- [x] #2 The red media_render_fixes pins that cover the More row are green with no assertion weakened
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run the two red More pins, record the clip.
2. Widen the More ItemGrid columns 16 -> 17 so the danger action's own 2-cell margin fits inside its column.
3. Add 60x24 coverage in Tests/UI/test_library_crit9_media_reader.py.
4. Live-verify at 235x52 / 100x30 / 60x24.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The More disclosure's ItemGrid used min_column_width=15/max_column_width=16. GridLayout.arrange resolves container_width = min(len(children), width // max_column_width) * max_column_width and then columns = container_width // min_column_width, so every action got exactly a 16-cell column. `.library-media-action-danger` (task-31980) adds `margin: 0 0 0 2`, taken out of the button's OWN box, leaving 'Move to trash' a 14-cell region for a 15-cell auto width -- Textual wrapped the label and only 'Move to' was visible at height 1. Widened the column to 17/17 so the longest label, the Button's own two auto-width cells and that 2-cell separation fit together; the TCSS margin rule is untouched, so task-31980's visual separation is preserved.

The two red pins (test_more_opens_one_row_and_moves_the_reader_body_by_one, test_more_stays_compact_at_the_narrow_reader_width) are green with no assertion weakened, as are the two other More pins. Added the 60x24 stage AC#1 names as a new painted-text test; confirmed it fails on the pre-fix 15/16 geometry and passes on 17/17.

Live at 235x52 one row reads '  Edit metadata    Open original    Open manager       Move to trash'; at 100x30 and 60x24 the grid reflows to two rows with all four labels whole.

Files: tldw_chatbook/Widgets/Library/library_media_viewer.py, Tests/UI/test_library_crit9_media_reader.py, Docs/User_Guide/library/media-and-conversations.md.
<!-- SECTION:NOTES:END -->
