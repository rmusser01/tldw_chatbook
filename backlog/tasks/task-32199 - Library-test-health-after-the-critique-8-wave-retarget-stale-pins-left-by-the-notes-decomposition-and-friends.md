---
id: TASK-32199
title: >-
  Library test health after the critique-8 wave: retarget stale pins left by the
  notes decomposition and friends
status: In Progress
assignee: []
created_date: '2026-09-10 14:19'
updated_date: '2026-09-10 15:12'
labels:
  - tests
  - library
  - test-health
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Several clusters of Library test failures were already red on dev before the critique-8 fix wave (PRs #2519 #2523 #2524 #2525 #2528 #2531 #2533) started, caused by earlier dev changes that never updated their tests: a hardcoded diagnostic-inventory registry invalidated by a log-call move and two retired diagnostics, a SimpleNamespace fake missing an attribute dev added to the conversations clear-filter handler plus a layout-timing read in its sibling, a media-render capture helper that stopped stripping the focused row's new border bar together with two stale Items-pane width pins, and a screen-size ratchet whose pins sit below the current file sizes. None was fixed inside those PRs, so every fix wave paid the same baseline tax of re-diagnosing red tests it did not cause. This task clears what can be cleared and records the rest for an owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed test is green on dev or explicitly recorded as needing an owner decision
- [x] #2 No production behaviour changes except where a test proved the code wrong, each named in the notes
- [x] #3 A whole-file run of each touched test file shows no new failing names versus the dev tip
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm each item is red at the dev tip (02374bf66a) before touching it, capturing the failing names per file as the baseline.
2. Diagnostic inventory: re-point the hardcoded registry at the tree (a moved log call, two retired diagnostics, one that deliberately gained a traceback) and add the summary key the generator now emits.
3. Conversations clear filter: give the SimpleNamespace fake the browse-scope attribute the handler now branches on; fix the sibling's layout-timing read while in the file.
4. Adaptive reader pin: verify at the dev tip before assuming it is red.
5. Media render captures: find and fix the common causes -- the focused row's border bar bleeding into captured text, and stale Items-pane width pins -- and trace whatever remains to a named cause.
6. Screen-size ratchet: read the ratchet's own policy and do not re-pin against it.
7. One commit per item with explicit paths, a RED run before and a GREEN run after, plus a whole-file run of each touched file compared with the dev tip by NAME.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Cleared what was clearable of the pre-wave Library baseline; the rest is traced and named below rather than papered over. Three test files touched, no production change in this branch.

**Item 1 (notes-row ids) -- NOT IN THIS BRANCH.** Owned by task-32175 (branch `fix/library-notes-r-tests`, which repairs the shell node ids and the notes canvas/reader/file-workspace files) and by PR #2564, which already landed the Notes-canvas reds on dev. An earlier commit here that retargeted four of those waits was dropped once that overlap was known; `Tests/UI/test_library_shell.py`, `test_library_entry_compose_once.py`, `test_library_notes_reader.py`, `test_library_file_notes_workspace.py`, `test_library_notes_canvas.py`, `test_library_notes_add_from_files_canvas.py` and `test_library_notes_wave_list.py` are untouched here.

**Item 2 (diagnostic inventory) -- DONE.** `test_reviewed_diagnostic_changes_are_metadata_only` hardcodes the owning file of each reviewed diagnostic and four rows had gone stale: "Failed to restore a Library note" moved into `library_notes_controller.py` with the wave-8 notes decomposition (9e13f0207c) so its row moved with it; the two "Console session startup: ..." diagnostics were retired by 11150b849d and their rows are gone, with the commit named; "canvas sync failed" deliberately gained a traceback in 51533602c4 (TASK-32089), so it is no longer a metadata-only diagnostic and leaves this registry -- it is not a persistent sink and the inventory manifest still owns it. `test_inventory_excludes_nested_virtualenv_but_keeps_application_sources` also pinned a summary dict predating the generator's `task_31551_calls` counter; the key is added. 69 passed, 1 skipped (was 3 failed).

**Item 3 (conversations clear filter) -- DONE.** The handler gained a `_library_unavailable_browse_scope` branch on dev and the `SimpleNamespace` fake never grew the attribute. Its sibling in the same file was red for an unrelated reason: select mode recomposes the list, so the state flag flips a frame before the rebuilt rows are laid out and `.region` read 0x0; it now waits for a laid-out row before computing click offsets. 20 passed (was 2 failed).

**Item 4 (adaptive reader shell pin) -- ALREADY GREEN.** `test_media_items_pane_grows_with_the_terminal_once_reader_is_comfortable[size0-56-46]` and the whole file pass unmodified at the dev tip (35 passed). Nothing to fix; the report of it being red did not reproduce.

**Item 5 (media render captures) -- PARTIAL, 4 of 12.** There was no single common cause. Fixed: `_painted_item_lines` returned the raw crop, so since task-31983 gave `.library-media-row:focus` a `border-left: thick` bar the row that entry focus parks on (task-2856) captured as `'█    document · 5m · analysed'` and stopped comparing with its own siblings -- one cell blanked on the way out, so every column index in the file still lines up; and `test_media_rows_paint_analysed_only_for_analysed_items` pinned a 52-cell Items pane at both sizes, stale since 9187bc0307 (task-31979) handed the empty Reader's width to the list (132 wide) and the below-64 stage work re-fitted the narrow one (44). Whole file: 8 failed / 115 passed (was 12 failed / 111 passed), no new names.

**Item 5 remainder -- DECISION-NEEDED (8 names, three causes).**
- `test_select_mode_entry_focuses_a_row_so_down_and_space_work[size0,size1]`: `_row_is_painted_focused` lives in `Tests/UI/test_library_shell.py` (frozen for task-32175) and asserts `all(style.underline)` over the row's painted cells -- the task-31983 bar cell is not underlined, so a genuinely focused row reads as unfocused. One-line helper fix, in someone else's file.
- `test_more_opens_one_row_and_moves_the_reader_body_by_one`, `test_more_stays_compact_at_the_narrow_reader_width`, `test_more_row_actions_share_one_grid_column_grammar[wide,narrow]`, `test_reader_focus_changes_border_glyphs_not_only_colour`: the Reader's More-row actions clip at the width the empty Reader is left with -- "Move to trash" paints as "Move to". Verified NOT to be task-31979's widening (disabling that block leaves them red). This looks like a real UX regression, not a stale pin, so it is left for an owner rather than re-pinned.
- `test_analysed_secondary_survives_the_36_cell_items_floor`: a custom `items_width=36` no longer reaches the pane (it stays at the automatic width) even with the task-31979 widening disabled, i.e. the custom-width path itself, not the pin.

**Item 6 (screen size ratchet) -- DECISION-NEEDED, deliberately not re-pinned.** Red at the dev tip: `library_screen.py` 33261 lines vs a 33204 budget (+57), `chat_screen.py` 23958 vs 16966 (+6992), and `test_task_22507_4_does_not_worsen_chat_screen_base` (23958 <= 20099). The ratchet's own failure message is the policy -- "Do NOT raise the budget to make this pass. Lower it when a decomposition wave lands" -- so no re-pin was made. Two open peer PRs (#2543, #2547) add further lines to `library_screen.py` without re-pinning, so any pin set now drifts on merge. The owner must choose between one deliberate re-pin after those land and a decomposition wave. AC #1 is left unticked for this.

Modified files: `Tests/Architecture/test_persistent_diagnostic_inventory.py`, `Tests/UI/test_library_multiselect_conversations.py`, `Tests/UI/test_library_media_render_fixes.py`. `./scripts/preflight.sh` passes.
<!-- SECTION:NOTES:END -->
