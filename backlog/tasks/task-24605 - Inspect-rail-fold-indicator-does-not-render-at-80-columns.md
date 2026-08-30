---
id: TASK-24605
title: Inspect rail fold indicator does not render at 80 columns
status: Done
assignee:
  - '@claude'
created_date: '2026-08-30 00:54'
updated_date: '2026-08-30 02:46'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The fold hint renders at 235 and 120 columns but never at 80, where measurement showed only 2 of 11 sections visible and 9 below the fold. At maximum scroll the last visible content was 'Artifacts: Connected -' clipped mid-sentence with no closing border and no hint. This is the exact failure the fold-indicator convention exists to prevent. Fixed non-scrolling chrome is 8 rows, 62 percent of the usable rail height at that size, so the hint row loses the space contest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The fold hint renders at 80x24 whenever Inspect rail content overflows
- [x] #2 Rail content is never clipped mid-sentence as the only signal that more exists
- [x] #3 Fixed non-scrolling chrome is reduced at narrow widths so scrollable content is not a minority of the rail
- [x] #4 A test asserts hint presence at 80x24 with overflowing content and absence at scroll end
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The fold hint IS rendered at 80x24 now, verified live. The cause was not the predicate.

What the critique reported: no fold hint at 80 columns while 9 of 11 sections sat below the fold, with content clipped mid-sentence. Reproduced live twice.

What it actually was: '#console-right-rail' declared 'min-height: 20'. On a 24-row terminal the rail is allotted 13 rows, so it laid its children out against 20 rows and was then clipped to 13 -- and the child that fell off the bottom was the fold hint, because it is the LAST one. The hint reported display=True throughout. 'display' is not 'visible', and every layer above the paint said the hint was fine.

outer_hint_required and the reconcile were correct all along and are untouched. The floor is now 12, below what the smallest supported terminal actually yields, so the rail sizes to its real box.

Verification honesty: the Textual test harness does NOT reproduce this -- it allots the rail enough rows that nothing clips, so the regression test passes both before and after. The evidence is a live 80x24 tmux run: before, the body overflowed (scrollbar painted) with no hint anywhere; after, '▼ more sections — scroll' paints on the rail's last row. The test was still strengthened to assert the hint's REGION lies inside the rail's box rather than merely that display is True, since that is the property that was false; it guards the class of defect even though it cannot currently fail on it.

Modified: tldw_chatbook/css/components/_agentic_terminal.tcss (+ regenerated bundle), Tests/UI/test_console_narrow_layout.py.

Also in that file: _open_inspector_narrow now opens the rail with alt+i rather than clicking the handle, because below 84 columns the handle does not exist -- the click helper could not open the rail at 80 at all, which was TASK-24600.
<!-- SECTION:NOTES:END -->
