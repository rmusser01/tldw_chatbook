---
id: TASK-2701
title: >-
  Library Prompts editor: action row renders below the viewport (Save
  unreachable by mouse)
status: Done
assignee:
  - '@codex'
created_date: '2026-07-31'
updated_date: '2026-08-09 00:56'
labels:
  - library
  - bug
  - ui
dependencies: []
---

## Description (the why)

The prompts editor's action row (Save / Use in Console / Export… / Copy
text / Duplicate prompt / Delete) lays out one row past the bottom of the
screen and is clipped by its parent, so it never becomes visible and
cannot be scrolled to. Measured on dev @ 207053253 during the G3
user-guide session (2026-07-31): at `run_test(size=(200, 50))` the row's
buttons region at `y=50` (viewport rows are 0-49) inside a parent
`Horizontal` whose region ends at `y=49`; identical symptom in a real
terminal at 200×50 (full-pane capture shows fields, the "New prompt · •
Unsaved changes" meta line, then the frame — no action row), and the G2
guide capture `prompts-editor.svg` lacks the row for the same reason.
The buttons ARE composed and focusable — Tab reaches them and Enter
activates (blind save works) — so the feature is keyboard-only in
practice at standard heights. The notes editor's action row renders fine
at the same size, so this is specific to the prompts editor's layout
(the skills editor avoided the same trap by using a `VerticalScroll`
root).

Related: task-2703 is the same symptom family in the Console Edit
Message modal; a shared root cause is plausible.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Editor actions remain visible at 200x50.
- [x] #2 Actions remain scroll-reachable at shorter terminal sizes.
- [x] #3 The action area does not obscure the final editor field.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implemented in the TASK-202 PR; split the editor into a scrollable body and auto-height footer and add geometry regressions. ADR required: no; ADR path: N/A; reason: UI-only defect repair under ADR-011/ADR-040.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented in the TASK-202 PR. The Prompt editor now has one scrollable content owner plus persistent auto-height action groups, while the embedded structured block editor avoids a nested vertical scroll trap. Geometry regressions cover normal and conflict states at 80x24, 100x30, 140x40, and 200x50, asserting reachable actions, visible final fields, and no overlap; the post-rebase affected suite passed 145 tests and the Impeccable layout scan returned no findings.

ADR required: no. ADR path: N/A. This is a contained Textual layout correction under the existing Library UI architecture.
<!-- SECTION:NOTES:END -->
