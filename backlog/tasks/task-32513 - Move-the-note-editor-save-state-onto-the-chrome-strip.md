---
id: TASK-32513
title: Move the note editor save state onto the chrome strip
status: To Do
assignee: []
created_date: '2026-09-11 17:25'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - rider
dependencies:
  - TASK-32143
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-32143 shipped the editor chrome strip but could not give it the save
state, so the editor still reports saving on a line of its own above the mode
tabs — one word ("Saved") beside 130 empty columns at 235x52, which is what
the critique called floating. The strip under the body is where a terminal
reader looks for it.

The blocker is measured, not assumed. `apply_compact_presentation` sets the
save state's `width`, `height`, `min_height`, `max_height`, `text_wrap` and
`text_overflow` as INLINE styles that assume it lives in
`#library-note-header-second-row`, and pins that band's `min_height` to 3. So
(a) the widget cannot simply be reparented — its inline sizing follows it and
resolves to 3 rows in the strip at wide sizes — and (b) the band keeps 3 rows
with or without it, measured at 60x20, so the move costs the body a row at
every width (6 → 5 there), the row task-32217 spent a fix reclaiming.

That compact block is shared code pinned by peer PR #2605 and task-32360 at
60 columns, which is why task-32143 did not rewrite it in passing.

Evidence: `wave3-caps/chrome-strip/10-235x52-strip-on-open.txt` (the floating
line and the strip on the same screen);
`93-evidence-status-crushed-at-120x40.txt` (why the facts cannot share the
save state's row instead).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The note editor shows its save state on the chrome strip under the body, and nowhere else — no second save-state line at any terminal width
- [ ] #2 The editor body keeps the height it has today at 60x20, 80x24 and 100x30 (no row is lost to the move)
- [ ] #3 The compact save-state shapes peer PR #2605 and task-32360 pin at 60 columns still hold, with their tests green
- [ ] #4 Live-verified at 235x52, 100x30 and 60x24 with captures, showing the save state changing (dirty → saving → saved) on the strip
<!-- AC:END -->
