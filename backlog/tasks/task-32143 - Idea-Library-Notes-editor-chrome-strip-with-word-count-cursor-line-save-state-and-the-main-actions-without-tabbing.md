---
id: TASK-32143
title: >-
  Idea: Library Notes editor chrome strip with word count, cursor line, save
  state and the main actions without tabbing
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-08 21:39'
updated_date: '2026-09-11 17:11'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - idea
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Improvement pitched by the design assessor for the power user: a status strip at the bottom of the body (where terminal users look) carrying words, cursor line and save state, plus Save / Preview / Keywords / Delete reachable without Tab. Today 'Saved' floats above the mode tabs and the word count lives under Info. Size S. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design agreed with the user before implementation
- [ ] #2 The strip replaces the floating meta line rather than adding to it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Live-reproduce the editor chrome at 235x52: confirm which line 'floats' (AC#2's target).
2. Read every pin on the candidate widgets before moving anything.
3. RED test on the real editor route: the strip's facts cell is absent.
4. Add the facts cell to the existing save-state row (the floating line becomes the strip), fed by the word count the controller already computes plus the body TextArea's cursor.
5. Hide it below 80 columns; never focusable.
6. GREEN; live captures at 235x52 and 100x30 and below 80; guide + stamp; CSS bundle sync + boot-CSS budget.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped the strip; AC#2 is NOT met and needs a ruling -- both ACs are left unticked deliberately.

**What shipped.** One right-aligned Static (`#library-note-chrome-facts`) directly under the note body, reading `N words · L:C` with a one-based caret. Shown only while Edit is the open view on a terminal >= 80 columns; Preview and Info have no caret and a narrower terminal gives the row back to the body. Never focusable, no new colours (muted to the same tier as the meta line).

**State feed, no new cost.** `_library_note_presentation_state()` already counts the words for `metadata_line` on every body change; the count now rides the presentation state as a number (`LibraryNotePresentationState.word_count`), so the strip adds no scan and no database read. The caret comes off the mounted TextArea through a `TextArea.SelectionChanged` handler on the canvas, because arrow keys never reach the presentation state. An `@on(Resize)` handler re-decides the 80-column gate (the compact flag only flips at 120).

**AC#2 -- 'the strip replaces the floating meta line rather than adding to it' -- not done.** The save state ('Saved' alone on a line above the mode tabs at 235x52) stays where it is, so the strip ADDS a line rather than replacing that one. Three measured reasons:
1. `apply_compact_presentation` sets the save state's width/height/wrap/overflow as INLINE styles that assume it lives in `#library-note-header-second-row`, and pins that band's `min_height` to 3. Moving the widget therefore means rewriting that shared compact block -- which peer PR #2605 and task-32360 pin at 60 columns.
2. Because the band keeps min_height 3 with or without the status, moving it frees no row: the strip becomes +1 row at EVERY width, including 60x20, where the body drops 6 -> 5 (measured). task-32217 spent a whole fix making that body take the pane.
3. Putting the facts ON the existing status row instead (no new row) was tried and reverted: measured live, the row has no slack. At 120x40, 160x45 and 190x45 the save state was crushed to a single character ('S5,408 words · 1:36') because `#library-note-primary-actions` resolves ~93 cells wide whatever the pane is. Capture: wave3-caps/chrome-strip/04-120x40.txt.

**Deviations from the wave-3 plan's design line.** (a) No accelerator on the strip: Library Notes deliberately has NO save accelerator -- `Tests/UI/test_library_honesty_accessibility.py::test_notes_ctrl_s_is_absent_from_binding_footer_and_f1_while_skill_keeps_it` and `test_library_shell.py::test_library_note_ctrl_s_is_unavailable_while_save_remains_explicit` pin its absence, and the one real editor binding (`esc` back to notes) is already in the footer one row below. (b) No save state on the strip, per AC#2 above.

**One geometry re-pin.** `test_library_note_compact_surplus_allocation_expands_only_named_owner[editor]`: the strip is a fixed row at >= 80 columns, so the 1fr body is one shorter at 80x24 and 100x30 (10 -> 9, 16 -> 15). The strip is added to that test's fixed-selector list, so it still pins exact heights and still asserts only the named owner grows, by the same 6. Below 80 the strip is hidden and the four 60x20 allocation tests are untouched.

**Files.** `tldw_chatbook/Widgets/Library/library_notes_canvas.py` (module helper `library_note_chrome_facts`, `NOTE_CHROME_FACTS_MIN_WIDTH`, the compose line, `update_note_chrome_facts` + two handlers, the `word_count` state field), `tldw_chatbook/UI/Library_Modules/library_notes_controller.py` (one line: carry the count it already has), `tldw_chatbook/css/components/_agentic_terminal.tcss` (rule splits into the screen-owned sheet; boot bundle unchanged), `Tests/UI/test_library_notes_wave_chrome_strip.py` (new, 5 tests), `Tests/UI/test_library_shell.py` (the one re-pin), `Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
