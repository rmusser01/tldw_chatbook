---
id: TASK-42473
title: "Library Media's focus handler still disarms list-entry focus when Textual's focus reset lands on a pane grip"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: '2026-11-23 23:30'
labels:
- follow-up
- library
- focus
- hardening
priority: low
references:
- tldw_chatbook/UI/Screens/library_screen.py
- Tests/UI/test_library_rail_switch_keeps_reader_and_note.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42422. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** TASK-42422 fixed a Notes-to-resident-Media press that left Library focus on `#library-browse-library-grip`. A queued recompose removed the focused control, Textual's focus reset parked focus on the Library pane grip, and the screen's `DescendantFocus` handler disarmed the pending Media list-entry focus, because only Notes treats a grip landing as the app's own (Notes is exempt since TASK-41129). TASK-42422's guard keeps the entry's own landings off that path, so it left the handler alone (`UI/Screens/library_screen.py:10409`, `on_descendant_focus`). Any other route by which Textual's reset lands on a grip inside Media's entry window would still disarm the entry and leave focus on the grip, which is the defect class of TASK-41504. No failure has been seen since TASK-42422's fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A grip landing from Textual's focus reset inside Media's list-entry window does not disarm the entry, as for Notes; a test drives such a landing (for example by removing the focused control mid-window) and fails on the current handler
- [x] #2 A user's own move to a grip (a click or a key) still disarms the entry, for Media and Notes alike; a test shows it
- [x] #3 The rail-switch first-row tests pass in repeated runs under parallel load (at least 10 runs, or a soak recorded in the notes)
<!-- AC:END -->

## Implementation Plan

1. Read `on_descendant_focus`, `get_focusable_widget_at`, `on_key` and TASK-42422's guard; list every route by which focus can land on a pane grip inside Media's entry window.
2. Test first: drive Textual's reset onto a grip mid-window on the Media route (remove the focused control); it must fail on the current handler.
3. Treat a grip landing as the app's own for Media as Notes does, and make a click on a grip disarm before Textual focuses it on Media as it does on Notes.
4. Test that a click and a key move to a grip still disarm, for Media and Notes.
5. Soak the rail-switch first-row tests 10+ times at `-n 4`; raise the `library_screen.py` ratchet by the exact growth.

## Implementation Notes

**Round 24 (2026-11-23, branch fu/r24-library, base dev b31e3e9241).**

**Routes to a grip inside Media's entry window** (probe at 170x48, production stylesheet):
- Textual's reset: a Media canvas repaint that removes the focused `type:` control while the entry waits for the repaint's rows parks focus on `#library-browse-library-grip`. The handler disarmed the entry, and focus stayed on the grip. This is the defect.
- A user's click or Tab onto a grip (Tab from the last row reaches `#library-browse-items-grip`). `on_key` disarms first for keys. A click on Media is disarmed by the focus handler itself.
- Media also focuses grips itself (the resize hand-off to the Items grip), and tests focus grips in code.

**Fix (`UI/Screens/library_screen.py`).**
- `LibraryScreen._reset_focus` calls Textual's reset unchanged and records where it landed (`_library_reset_landing`).
- `on_descendant_focus`: on Media, a grip landing is the app's own (`grip_reset`: clear focus, keep the entry armed) only when it is that recorded reset landing and the entry has not yet landed a row since it was armed (`_library_entry_row_landed`, set when the armed list's row takes focus, cleared on arm). Notes keeps its TASK-41129 rule (any grip). A Media return receipt is no longer revoked by such a landing.
- **Why both conditions.** A first cut that exempted every Media grip landing, as Notes does, failed 9 existing Media tests that passed at base. Some focus a grip themselves, and in some the entry yanked focus to row 0 later (bulk-delete Undo, the compact Trash pane). Each condition has a pin among them:
  - Without the recorded-reset check: `test_media_trash_compact_status_folds_after_two_readable_rows` and `test_media_trash_compact_pane_priority_survives_page_and_filter_refresh` fail ("Compact Items pane never opened": their own `grip.focus()` was cleared).
  - Without the not-landed check: `test_full_success_bulk_delete_focuses_undo_and_enter_restores` fails (`'library-media-row-0' == 'library-media-bulk-delete-undo'`).
  - Once a row has landed, repaints restore focus by identity, so the window's original grip rule stays.
- The lesson's warning about overriding `Screen._reset_focus` (TASK-34000.28) concerned changing where the reset lands. This override does not change it; if Textual renames the method, the AC#1 test goes red.

**Red/green.**
- AC#1, `test_a_reset_onto_a_grip_inside_the_media_entry_window_keeps_the_entry` (rail-switch file): it focuses `type:`, re-arms the entry, holds the repaint's rows and recomposes the Media canvas. Base: 4 of 4 red, "The grip landing disarmed the entry (LibraryAdaptiveReaderPaneGrip(id='library-browse-library-grip', ...))". Fixed: green; nothing is focused while held, then focus is on row 0.
- AC#2, `test_a_users_own_move_to_a_grip_disarms_the_list_entry[click|tab x media|notes]`: green at base (16 of 16) and fixed; the entry disarms and focus stays on the Items grip.

**The retry node (shared with TASK-41723).** `test_library_media_initial_error_is_unknown_and_retry_is_unique` presses Retry with `press()` inside the Media entry window. At base, the Retry's removal reset focus onto the Library grip, and that landing disarmed the entry, which let the test pass. The entry had only landed on Retry, not a row, so with this fix it survives and lands on row 0 when the retried rows arrive. With the first cut: 4 of 40 failed at `-n 4` (load 39-41), "focused='library-media-row-0'". A traced probe showed the same row-0 shape at base too (3 of 24), when the press beat the entry's own Retry landing. A real Enter or click disarms the window first (`on_key`, `on_mouse_down`). So the test now waits for the window to close before its code-driven press, as TASK-42422 did for the note test; its assertions are unchanged. After: 40 of 40 at `-n 4` (load 50-62).

**Runs (final tree).**
- AC#3: the parametrized rail-switch first-row test (12 cases) plus the two new tests (5 cases), 10 runs at `-n 4`: 170 passed (load 31-43).
- Covering run: the 24 Tests/UI files that reach the entry channel or the grips, plus every `test_library_media*.py` (44 files), at `-n 4` with a temporary local plugin that adds the `bootstrap_profile` marker (as TASK-41504 did; deleted), load 31-61. 29 failed and 2115 passed. Every one of the 29 also fails at base (screen and canvas reverted, same command): the 25 from the first covering run, then the 7 left of 14 new ones on rerun. That run was one commit earlier, before the reset record was made one-use (which only narrows the exemption). After that change, the 25, the rail-switch file and the retry node were rerun: nothing failed that passes at base.
- The whole rail-switch file (27 tests) passed in that final rerun.

**Size.** `library_screen.py` 32866 -> 32901 lines, 1288 -> 1289 methods (the `_reset_focus` override); the ratchet row is raised by exactly that, with a dated note. No timer added, so no clock-root row.

**Docs.** No User Guide change: the documented landing (the list's first row) is unchanged; the fix removes a race.

**Files.** `tldw_chatbook/UI/Screens/library_screen.py`, `Tests/UI/test_library_rail_switch_keeps_reader_and_note.py`, `Tests/UI/test_library_shell.py`, `Tests/Architecture/test_screen_size_ratchet.py`.

**Checks (final tree, shared by TASK-41723, 42470 and 42473).** `Tests/CI` at `-n 4`: 1823 passed, 2 skipped. `Tests/Architecture` serially: 1491 passed, 5 skipped, 1 failed. The failure was `test_persistent_diagnostic_inventory.py::test_task_15103_review_ledger_canonical_schema_and_arithmetic`, at its 300 s timeout while the run shared the host at load 33-67. Alone it passed in 194 s (load ~45-50). `./scripts/preflight.sh` passed; `ruff check tldw_chatbook --select F` passed; `Tests/lint/ruff_gate.py` passed (RATCHET OK).
