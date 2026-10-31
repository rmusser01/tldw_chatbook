---
id: TASK-33787
title: >-
  Roleplay: the rail search box hides what you type once focused, and the list
  below jumps a row
status: Done
assignee: []
created_date: '2026-10-02 04:56'
labels:
  - roleplay
  - ux-review-2026-10-01
  - bug
  - layout
dependencies: []
references:
  - Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md
  - Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/findings.json
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the 2026-10-01 Roleplay NN/g + HCI review: finding RP-009 (P0, severity 3, effort S). Report: `Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md` (section 5 and Appendix A), with the machine-readable entry in `findings.json` beside it. The review ran on origin/dev @ 84247cb843 (worktree `.worktrees/roleplay-library-ux`).

Line refs below were re-checked on origin/dev @ ab4df99959. Paths are under `tldw_chatbook/` (`personas_screen.py` is `UI/Screens/personas_screen.py`; the `personas_*` and `persona_*` widgets are in `Widgets/Persona_Widgets/`). Capture names refer to the review's capture set; those copied into the QA folder are marked `evidence/`.

**What happens.** Clicking or pressing Ctrl+F into the Characters search box collapses it to an empty two-line frame: the list filters as you type, but the query cannot be seen, checked or corrected. Focusing it also pushes every list row down one row, and blurring moves them back. This affects Characters and Personas at every measured width, and all four modes at 120x36. Every F6 into the rail lands on the search box and triggers the same reflow.

**Who it hurts.** Finding a card by name, the core of J1, is done blind. Typos and leftover text are invisible, and because a no-match search says the library is empty (RP-022), a typo reads as data loss.

**Cause, for context.** When the rail is too narrow for its toolbars it stacks them and forces the search box to one row with no border, so the last toolbar action (Tag) still fits inside the pane. The app-wide Input focus rule then adds a two-row border back, which leaves zero text rows. Whatever the fix, the Tag action must still fit.

**Evidence:**
- `personas_library_pane.py:155-164` (the stacked-mode override) and `:242` (the stacking trigger); `css/components/_forms.tcss:105-111` (the Input focus border). The same trap for compact inputs is documented at `_forms.tcss:113-130` (TASK-17961).
- Captures: `evidence/roleplay-characters-search-nomatch-nothing-selected-160x45.txt` rows 15-16; `evidence/roleplay-characters-search-typed-invisible-80x24.txt`; `review-rv-layout-search-focused-typed-160x45` (first row moved from y23 to y24); `review-vf-fit-search-typed-120x36`.
- Library precedent: its rail search is a stable 3-row box with a clear button (`library_rail.py:1099-1126`).

**Out of scope:** where F6 lands in the rail (RP-025) and the no-match copy (RP-022).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 In all four modes at 80x24, 120x36 and 160x45, the focused search box shows the typed query in at least one visible text row.
- [x] #2 Focusing, typing in and leaving the search box do not move any other rail widget: the list rows keep their screen positions.
- [x] #3 The search box has the same height focused and unfocused.
- [x] #4 At the widths where the rail stacks its toolbars today (including 80x24 and 120x36), the last toolbar action (Tag) still lies inside the pane.
- [x] #5 Regression tests assert at 120x36 and 160x45 that the focused search box has at least one content row and that the first list row's position is the same before focus, while focused and after typing; they fail on the current code.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- 2026-10-29 (Roleplay frame B2b): absorbed by TASK-33910.4 (spec 5.5, S): still To Do on dev at B2b's start, so its AC#1-#5 are TASK-33910.4's AC#19-#23, delivered by B2b's one-row toolbar (Task 7) and the placeholder lift (Task 8). It stays To Do until B2b's close-out sets it Done with the evidence.

- 2026-10-30 (Roleplay frame B2b, TASK-33910.4): delivered by B2b's one-row toolbar (fbb2260768), as TASK-33910.4 AC#19-#23 (spec 5.5, S, absorbed). The search box is one row high at rest and focused, so the typed query always shows and nothing below it moves; Tag and every other control lie inside the pane at every size. Pinned by Tests/UI/test_roleplay_item_list_app.py::test_the_focused_search_shows_what_you_type_and_nothing_moves[80x24|120x36|160x45] (all four modes per size: AC#1, #2, #3 and #5) and ::test_every_toolbar_control_is_whole_and_inside_the_pane[80x24|120x36|160x45|220x55] (AC#4). On dev 38623632e9 they fail (B2b's dry run: the box's rows went from (1, 1, 18) to (2, 0, 19) on focus at 80x24, 120x36 and 160x45). The focused placeholder paints at AA too (B2b Task 8).
<!-- SECTION:NOTES:END -->
