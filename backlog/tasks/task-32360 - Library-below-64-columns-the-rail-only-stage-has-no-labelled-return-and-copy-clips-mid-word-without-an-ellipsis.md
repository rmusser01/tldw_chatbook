---
id: TASK-32360
title: >-
  Library below 64 columns: the rail-only stage has no labelled return and copy
  clips mid-word without an ellipsis
status: In Progress
assignee: []
created_date: '2026-09-11 06:18'
updated_date: '2026-09-11 07:45'
labels:
  - library
  - layout
  - critique-10
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 60x24 on the rail-only stage Escape is the only return and the footer never says so; strings clip without an ellipsis; reader-shell grips overpaint list rows (B D8 caps 56-59; PROVEN library_adaptive_reader_shell.py:151-156). Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The footer names the return on every single-stage surface
- [ ] #2 Clipped copy ends in an ellipsis
- [x] #3 Grips never overpaint content
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Footer: return a SHORT context on the narrow stage so the return chip survives the tier.\n2. text-overflow: ellipsis on the canvas classes that clip.\n3. Measure grip regions at 235/100/60 before touching geometry.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two of three ACs closed; AC#2 left open with its measurement.

AC#1 (footer names the return) was ALREADY TRUE and is now pinned as a regression. The plan expected the chip to be lost to AppFooterStatus's width ladder and prescribed trimming the narrow-stage context to two chips; driving the real footer widget at width 60 with one, two, three and five chips renders 'esc back to Library | F1 · F6 · Ctrl+P · Ctrl+Q' in EVERY case, and the live app at 60x24 on a Library browse list paints exactly that. task-32225's 'FIRST, not appended' already carries it, so the prescribed trim would have dropped chips the footer was not painting anyway -- a no-op dressed as a fix, and it was reverted. Whatever B captured at 60x24 was a surface where this context is not active at all (the predicate stands down when an earlier Escape action owns the key, and requires a CLOSED Library pane).

AC#3 (grips never overpaint) was ALSO already true, by two independent measurements: `grip.region.right <= neighbour.region.x` holds at 235x52, 100x30 and 60x24, and a painted-frame scan finds no '<---'/'--->' run on any column outside a grip's own columns. B's '17-media-list.txt' runs at char columns 42, 42 and 183 ARE the grips' own columns (x=41 and x=182) -- the arrows looked stray because the handle was unlabelled, which task-32355 now fixes. Both assertions are pinned.

AC#2 (clipped copy ends in an ellipsis) NOT DONE, reproduced and narrowed. Live at 60x24 on Notes: the status line paints 'Library notes · Ready · Next: / Create a note or add from' and loses 'files.' (a HEIGHT clip -- the Static resolved two lines for three lines of content), and the toolbar paints 'New  Sort: Newest  Sel' (a WIDTH crop of the Button by its container). `text-overflow: ellipsis` on `.library-canvas-action` and `#library-notes-status` was tried and measured live: NO EFFECT on either, because neither loss is the horizontal text-overflow the property governs. The existing `#library-shell-grid.library-notes-compact` rules already carry the right fix for both (`width: auto` on the actions, `height: 1` + `text-wrap: nowrap` + `text-overflow: ellipsis` on the status) -- they are simply not in effect at 60x24 on the Notes route, so the real defect is the compact gate in `library_notes_controller.py` (`legacy_compact`), a file this branch does not own. Needs its own task against the Notes canvas.

Files: Tests/UI/test_library_crit10_layout.py.
<!-- SECTION:NOTES:END -->
