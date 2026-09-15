---
id: TASK-32613
title: >-
  Library Notes: a third of tab stops paint no focus, the stop count is unstable
  across modes, and Tab plus Enter from Preview exits the note
status: In Progress
assignee: []
created_date: '2026-09-15 06:40'
updated_date: '2026-09-15 17:59'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor B D17, D8 and K11, personas Sam and Alex, edit workflow.

What happened. B enumerated the stops by pressing Tab and grepping every capture for a focus glyph: the notes-list toolbar with an active filter shows no indicator at stops 5 and 7 of 8 (B cap 10), the sync configuration form shows none at stops 1-4 (B section 4), the Session Git commit form shows none at stops 1-4 (B cap 43). Roughly a third of all stops. This sits beside a genuine strength that must be protected: the indicators that do exist are shape-based -- buttons as bars, list rows as a leading block, text areas as a thickened border, inputs as a full box -- and all four survive a monochrome dump.

Two behaviours compound it. From Preview, the obvious way to Info (Tab then Enter) exits the note entirely, returning the right pane to 'Select a note to edit it here' with no warning (B cap 15, K10). And the toolbar's tab-stop count is not stable across mode switches: the identical Shift+Tab x5 that reached Preview once landed on nothing the next time (B cap 17, K11).

Cause INFERRED for all three -- focus order left to the framework's defaults; not traced. Wave 4's task-32537 (#2683) added Preview's footer chip but did not change the tab order, and task-32550 fixed a different tab-count complaint on the filter.

Adjacent: the Info tier's missing chips are filed separately; this task is the indicator and the order.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every tab stop inside the Notes surface paints a focus indicator that survives a monochrome capture
- [ ] #2 The number of Tab presses from the body to a given control does not change with filter state or previous mode
- [x] #3 Tab from Preview reaches the next editor control, and no single Tab-plus-Enter from a reading mode closes the note without warning
- [x] #4 A test enumerates the stops in each pane and fails when one has no indicator
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Enumerate the note work pane's tab stops per mode in a harness and find the ones with no focus rule.
2. Give the reading regions (Preview, Info) a forward-Tab target that is not the exit button.
3. Pin the stop enumeration so a chip-less/indicator-less stop fails the suite.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three findings, and the cause of two of them is the same fact: **Textual orders the focus chain by SCREEN POSITION, not compose order** (`Widget._focus_sort_key` = (y, x) of the virtual region, applied per container level in `Screen.focus_chain`). So the heading's '‹ Notes' is the note pane's first stop however the canvas is composed, and a reading region -- the bottom of the pane -- is its last. One Tab wrapped the cycle from the document straight onto the exit, and the obvious Tab-then-Enter closed the note. Reproduced headlessly before any change: Tab from `#library-note-preview-region` focused `#library-note-back`. There is no tab-index in Textual, so the ring is re-cut in `_move_library_screen_focus`: a forward Tab out of either reading region goes to `#library-note-edit`.

Trade-off, recorded rather than hidden: Back is consequently reached by Shift+Tab (one press from Edit) or Escape rather than by wrapping forward. Cutting the ring anywhere else would need a per-control override table; a `ponytail:` comment on the constant names the upgrade path.

AC#1 was measured, not inspected. Every stop of the navigator, the editor, Preview, Info, the lasting-sync canvas and the Session Git commit form was focused in turn and its computed (text-style, outermost edge type) pair diffed against its blurred pair -- outline first, because an outline is painted over the border's own cells, and colour dropped, because a monochrome capture drops it. Exactly two stops in the whole Notes surface changed nothing: `#library-note-preview-region` and `#library-note-context-region`, whose ':focus' rule swapped an accent colour into a border that was already `solid` while the reset's `*:focus { outline: solid }` repainted the same glyphs. Both now take `border: heavy` + `outline: none` -- still one cell, still non-obscuring, and legible without colour. The sync configuration form's stops and the commit form's stops all already painted; the commit form only looks cue-less under a panel-only harness, which does not load the app stylesheet and therefore misses `Input:focus` (measured both ways -- with the bundle the fields go `round` -> `solid`).

AC#2 is met for the part that is achievable and is recorded honestly for the part that is not. Pinned: the pane's first six stops are the same controls in the same order in Edit, Preview and Info, so a control's distance from the mode strip does not depend on the mode you arrived from. NOT met, and not a defect: the absolute count from the CONTENT differs between Edit (title + body) and a reading pane (one region), and the list toolbar gains a stop when a filter is set because 'Clear filter' is composed only then -- composing a dead control to keep a count constant would be the dishonesty this critique round is about.

Files: UI/Screens/library_screen.py, css/components/_agentic_terminal.tcss (+ generated sheets), Tests/UI/test_library_notes_w5_kbd_focus.py, Tests/UI/test_non_obscuring_focus_contract.py (the reading-region block's solid->heavy pin), Docs/User_Guide/library/notes.md.
<!-- SECTION:NOTES:END -->
