---
id: TASK-38710
title: "File Notes: the Offline folder row at 50x20 raises ValueError and can crash the Library screen"
status: Done
assignee:
- '@claude'
created_date: 2026-11-04 00:43
updated_date: 2026-11-04 01:44
labels:
- library
- file-notes
- bug
priority: high
references:
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
- Tests/UI/test_library_file_notes_workspace.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-36513 review on 2026-11-03. In a CSS-true `run_test(size=(50, 20))` harness, the File Notes workspace with its folder Offline (with or without a warning) raises `ValueError: range() arg 3 must not be zero` from Rich's `chop_cells` during a layout pass. The "More file actions" label is laid out at width 0 inside a grid. It reproduces at eba3ad14b5 and later.

Textual reports an exception raised during layout as an app error, and the app exits. In the real app that takes the Library screen, and the whole TUI, down at a terminal size a user can reach by resizing. The fix has to find which widget and which CSS produce the zero width, and fix it there.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 20 rows, no layout pass raises at any width from 40 to 235 columns (every column from 40 to 61, then a sample that includes the 79/80 narrow boundary and 235), in each folder-row state: linked, offline, offline with a warning, and pairing pending; both with no file open and with a file open in Manage
- [x] #2 With a file open in Manage, "More file actions" stays reachable at every swept width: shown, enabled, in the focus chain, focusable, on screen once focused, with its label painted
- [x] #3 A regression test sweeps those widths and states and is red on the base build
<!-- AC:END -->

## Implementation Plan

1. Reproduce in the CSS-true harness and walk the traceback to the widget and the rule that leave a zero width.
2. Write the width sweep (four row states, navigator view and a file open in Manage) and show it red.
3. Fix the measurement at its source: the one-row File Notes action buttons must not ask Rich to wrap their label.
4. Run the sweep green, plus the covering File Notes tests; record red and green evidence.

## Implementation Notes

**Root cause.** The widget is `#file-notes-maintenance-toggle` ("More file actions"); any `.file-notes-toolbar` button can do the same. The chain at 50x20 with no file open:
- Below 80 columns the narrow navigator view resolves items 32 + two 5-cell grips, so the work pane gets the remaining 8 cells.
- The Offline row costs the body a row. The edit and manage regions drop to one row, and both `#file-notes-editor-pane` and `#file-notes-manage-region` (`overflow-y: auto`) reserve a 2-cell scrollbar. That leaves 4 cells.
- `-stack-editor-actions` / `-single-editor-actions` turn the toolbar into a one-column `1fr` grid. Textual's grid measures a row's height with `get_content_height(column_width - gutter)` and ignores `min-width`. So the button's `padding: 0 1` leaves 2 cells, Textual's Button `line-pad: 1` takes both, and the label wraps at width 0. In `Content._wrap_and_format`, Rich's `chop_cells(text, 0)` then raises. Only exactly 0 raises, because a negative width makes an empty `range`. That is why it hit 46, 48 and 50 columns and not their neighbours.
- The reflow runs on the app's own layout pass (`Screen._refresh_layout` -> `Compositor.reflow`), so the app records the error and exits.

**Fix.** The one-row action-button rule (`.file-notes-toolbar Button, #file-notes-back, #file-notes-session-changes`, already `height: 1`) adds `text-wrap: nowrap; text-overflow: ellipsis`. Textual's truncate path is defined at every width, and a one-row button only ever showed its first wrapped line anyway. A narrow label now reads "More fi…" instead of "More". Two alternatives were rejected. A `min-width` floor cannot help, because the grid's row measurement ignores it. Hiding the sliver work pane is a shared-resolver change, and other widths and heights could still line the scrollbars up.

**Test.** `test_file_notes_folder_row_states_lay_out_at_every_width_40_to_235` (in `Tests/UI/test_library_file_notes_workspace.py`) covers the four row states. In each it sweeps 40-61 and then 64-235 (including 79/80/81), first in the navigator view with no file, then with a file open in Manage. In the second sweep it also checks that "More file actions" is shown, enabled, in the focus chain, focused, on screen and painted.
- Red on base: all 4 states raised `ValueError: range() arg 3 must not be zero` at 46 or 48 columns.
- Green: 4 passed (`-n 4`, load average ~37 on 18 CPUs).
- An uncommitted wider probe also passed (15 passed): three states, heights 12/16/24/30/44, widths 40-81 plus 100/120/160/235, in the navigator, Manage and Edit views.

**Files.** `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py` (CSS), `Tests/UI/test_library_file_notes_workspace.py`, `Tests/Architecture/test_module_size_ratchet.py` (9593 -> 9599, dated).

**Fold (2026-11-04, the coordinator approved).** `test_file_notes_root_row_keeps_state_word_beside_pairing_review_at_40_columns` flaked under load, and the cause was a product race, not its wait. TASK-36513's deferred fit (`call_after_refresh`) could run before the layout that seats "Review…". It then fitted 18 cells, the status became 9, and nothing fitted again.
- **Fix.** `#file-notes-root-status` is now `_FolderRowStatus`, which calls the fit again from its own `on_resize` when its width changes. It skips an unchanged width and the auto-width `-empty-root` prompt. Layout fixes this width (`1fr`, one row, `nowrap`), so a refit cannot resize it and cannot loop.
- **Tests.** `_settled_root_row` can now wait, with `_wait_until`'s default deadline, for the painted state its caller asserts, and the 40-column test uses that.
  - New: `test_file_notes_root_row_refits_when_its_deferred_fit_beats_layout`. It runs every deferred fit before layout and counts fits to show the count stays the same over 10 idle frames. Red on ae0f24ecef in both params ("not fitted to its width").
  - Negative control: with the Offline-only fallback removed, the 40-column test's `[Offline-True]` case went red.
