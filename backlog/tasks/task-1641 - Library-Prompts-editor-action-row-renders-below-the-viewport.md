---
id: TASK-1641
title: 'Library Prompts editor: action row renders below the viewport (Save unreachable by mouse)'
status: To Do
assignee: []
created_date: '2026-07-31'
labels: [library, bug, ui]
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

Related: task-1620 is the same symptom family in the Console Edit
Message modal; a shared root cause is plausible.

## Acceptance Criteria (the what)

- [ ] The action row is visible in the prompts editor at 200×50 (and on
      shorter terminals it is reachable by scrolling).
- [ ] A geometry test pins the row inside the viewport/scrollable region
      so it cannot silently fall off again.
- [ ] The User Guide quirk in `Docs/User_Guide/library/prompts.md` is
      updated/removed to match.
