---
id: TASK-40293
title: "Entering the Library Media or Notes list never focuses its first row, and compact Media's empty-page return leaves focus nowhere"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-13 07:20'
created_date: 2026-11-12 19:36
labels:
- follow-up
- library
- focus
- keyboard
priority: medium
references:
- tldw_chatbook/UI/Screens/library_screen.py
- Tests/UI/test_library_shell.py
- Tests/UI/test_library_media_side_by_side.py
- scripts/library_shell_push_deselect.txt
dependencies:
- TASK-39130
- TASK-39152
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down: TASK-39152 and TASK-39130 left these tests red, unchanged, as product defects. Checked at the round-13 stack top 8b633d0101, 2026-11-12: the nodes below fail when run together (run.sh, `-n 4`, load ~13), with the failure quoted.

**What the user sees.**
- Pressing Enter on the Media or Notes rail row should put focus on the list's first row, so the arrow keys move through the list at once. Focus stays where it was instead: `library-ingest-top-button` for Media, `#library-browse-library-grip` for Notes at 170x48.
- On compact Media, a filter or delete that leaves an exactly empty page returns the canvas to its list by itself (80e4cd7056, task-32043 / task-32086). Focus then ends at `None`, so the keyboard has nowhere to land.

**Cause, from TASK-39152's and TASK-39130's notes.**
- Since de4d38b59a, `_library_focus_channel_owns_this_window()` (`UI/Screens/library_screen.py:8950`) reads "no rows mounted yet" as a settled empty page. The whole-screen recompose therefore restores the pre-press focus and disarms the entry arm. Making the predicate own the window while the arm is set made the first Media node pass whole. Notes is probably the same cause.
- The compact empty-page return's whole-screen recompose strands focus. TASK-39130 places the fix in the task-32086 reset branch of the browse apply: arm the list entry focus when the recompose strands focus.

**Nodes** (failures at 8b633d0101):
- `Tests/UI/test_library_shell.py::test_library_media_list_focuses_first_row_and_arrow_keys_move_it`: "entering the list must focus its first row" (`library-media-row-0` not focused).
- `::test_library_media_tab_away_survives_a_background_recompose_within_the_settle_window` and `::test_library_media_entry_focus_survives_three_chained_recomposes`: `library-media-row-0` not focused.
- `::test_library_notes_list_focuses_first_row_and_arrow_keys_move_it`: "entering the list must focus its first row" (`library-notes-tree-note-1` not focused).
- `Tests/UI/test_library_media_side_by_side.py::test_compact_media_viewer_back_empty_page_focuses_recovery_control`: "Empty viewer return did not focus the Import recovery action; focus is None."

The `test_library_shell.py` nodes are deselected from the `library-shell-dev` push job (`scripts/library_shell_push_deselect.txt`, TASK-39153); remove each line when its fix lands. The side-by-side node runs on no lane; TASK-39130 laned the file's other passing nodes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Entering the Media or Notes list from its rail row focuses the first row and the arrow keys move through the list, also when a background recompose lands inside the entry window
- [x] #2 On compact Media, the automatic return from an exactly empty page lands focus on a visible recovery control or list control, never on nothing
- [x] #3 The five nodes pass 3 of 3 at `-n 4`, each shown red before the fix, and their lines leave `scripts/library_shell_push_deselect.txt`
- [x] #4 The side-by-side node runs on a PR lane
<!-- AC:END -->

## Implementation Plan

1. Re-run the five nodes on the base (26b2e677dd) and trace focus writes, arm/disarm and the channel-ownership predicate during Media and Notes entry with a temporary probe.
2. Media: make `_library_focus_channel_owns_this_window` stand down for an empty page only once the page has settled empty (not loading, applied, no retained items), so a loading entry keeps the channel.
3. Notes: in `_focus_library_list_entry` (the seam every entry path uses), treat rows whose ancestors already have a recompose queued like pruning rows and retry, so Textual's reset off a replaced row cannot disarm the channel.
4. Compact empty-page return: arm the list-entry channel in the task-32086 reset branch when no focus identity was given, and let the empty page's one recovery action lead `type:` in `_library_media_empty_list_fallback_target`.
5. Show each node red with its fix reverted, remove the four deselect lines, lane the side-by-side node at the UI census tail.

## Implementation Notes

Premise held on base 26b2e677dd: all five nodes failed as filed. Traced with a temporary `set_focus`/arm/disarm probe (deleted).

- **Media entry.** On entry the page is still loading, so `_library_focus_channel_owns_this_window()` saw no rows and no fallback control (the Items pane is disabled until the shell opens it) and returned False; the whole-screen restore put focus back on the pre-press widget and that `DescendantFocus` disarmed the channel. The empty-page leg now also requires the page to have settled empty (`not loading`, `applied_result` set, no `retained_items`).
- **Notes entry.** The channel did land on `library-notes-tree-note-1`, but the Notes canvas's `apply_pane_width` had queued a `refresh(recompose=True)` a few ms earlier. The rows were not `_pruning` yet, so the recompose pruned the focused row, Textual's reset moved focus to the Library grip, and `on_descendant_focus` disarmed. `_focus_library_list_entry` (the seam every entry path uses) now treats rows whose ancestors have `_recompose_required` like pruning rows and retries after the recompose (read with `getattr`, like the `_pruning` check beside it: `test_screen_navigation.py`'s fake rows have no `ancestors`, and the first cut's direct attribute read failed `test_focus_library_list_entry_falls_back_to_first_row_outside_active_selection`). Lesson added to `backlog/docs/lessons-textual.md`.
- **Compact empty-page return.** The task-32086 reset branch's whole-screen recompose left focus at None; with no focus identity it now arms the list-entry channel. The channel's empty-page target (`_library_media_empty_list_fallback_target`) put `type:` first, so the page's one recovery action (Show all types / Import media / Retry) now leads and `type:` is last; a query in force still puts the filter first.
- Red/green, each with only that piece reverted (load 64-100): predicate -> the three Media nodes fail on `library-media-row-0.has_focus`; row skip -> the Notes node fails "entering the list must focus its first row"; reset arm -> the side-by-side node fails with focus None; fallback order -> it fails with focus on `library-media-type-filter`. All five pass with the fix (3 of 3 at `-n 4`, load ~41: the eleven stream nodes passed 11/11 in each of three runs).
- AC#4: the side-by-side node is appended at the `scripts/ui_pr_gate_census.txt` tail, `MINIMUM_FILES` 2028 -> 2029. Serial time with the UI lane flags (`-p no:xdist --timeout=180 --durations=0`): 15.6 s call, 31.9 s whole, load 55-62.
- Timing note: `test_library_media_tab_away_survives_a_background_recompose_within_the_settle_window` asserts the 2 s arm is still live after the landing. The Items pane opens only after the shell's first measure (1.0-1.6 s after the arm at load 28-65), so under very heavy load the arm can expire just before that assertion; one -n 4 run at load ~65 failed there before the full fix set landed. The 2 s ceiling is the existing `ponytail:` note's (task-32301).
- Files: `UI/Screens/library_screen.py`, `Tests/Architecture/test_screen_size_ratchet.py` (38167 -> 38183), `scripts/library_shell_push_deselect.txt` (4 lines), `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `Docs/User_Guide/library.md`, `backlog/docs/lessons-textual.md`.

Covering run (shared by the stream's five tasks, final tree): 58 Library UI files that reference the changed seams (`test_library_shell.py` whole, side-by-side, canvas-sync defects, choice strips, multiselect, honesty, workbench focus, screen navigation and 50 more), `-n 4`, load 42-66: 2023 passed, 863 failed. Every failing function was re-run on the head tree and on the base source (26b2e677dd, same tests): of those 871 nodes 850 fail on head and 845 on base, almost all the local RecoveryRequired wall for plain Tests/UI files plus the deselected nodes owned by TASK-32503/34000.60/40296/40299. Head-only reds re-checked: the fake-row unit test above (fixed), `test_any_non_markdown_item_with_content_says_why_rendered_is_absent[video]` (3 of 3 green on head when re-run alone), `test_literal_history_restore_keeps_kind_definition_and_guards` (red 1-3 of 3 params on base too), and four nodes that passed on re-run. `Tests/Architecture` Library wiring files: 49 passed (serial).

**Fix round 1 (review r14lf-rev1).**
- Critical: under load the reset branch's list-entry arm was revoked by Textual's queued DescendantFocus for `#library-browse-library-grip` (the Items prune parks focus there), leaving focus at None (reviewer: 3 of 12 probe runs). The arm is gone. `LibraryMediaController._land_library_media_emptied_page` now lands focus itself after the recompose: it resolves `_library_media_empty_list_fallback_target()` in a `call_after_refresh` and retries every 50 ms (up to 2 s) while the target is missing, pruning or not yet focusable. A queued foreign focus event has nothing to revoke.
- Important: the landing now takes only focus the reset strands: none, detached, a pane grip, or anything inside the Media browse shell outside `#library-rail`. Focus in the rail search box (or the nav bar) stays, as on base. It checks before the recompose and again at landing.
- Tests: the compact node now also asserts Import still holds focus 0.5 s after it lands. `test_compact_media_empty_page_landing_survives_a_queued_grip_focus` reproduces the ordering deterministically: it parks focus on the grip right after the reset and puts its DescendantFocus on the screen's queue. With the old arm restored it fails 2 of 2 with focus None; it passes with the fix. `test_wide_media_reader_reset_leaves_focus_the_user_holds_in_the_rail` (170x48, focus in `#library-search-input`) fails with focus on `library-media-empty-import` under the old arm and under the landing without its stranded gate, and passes with the fix.
- `-n 4 --dist each` (4 copies of each node): the compact node and the wide node passed 8/8 in each of 4 rounds (32/32, load ~17). The old arm also passed 24/24 at that load (the race needs more), which is why the queued-grip test pins the ordering itself.
- Both new nodes are appended at the UI census tail (`MINIMUM_FILES` 2029 -> 2031). Serial times with the lane flags: 7.3 s and 8.4 s call, about 15 s whole, load 19-36.
- The User Guide sentence now says the return lands only focus that was in the Media panes.
- Size rows: `library_media_controller.py` 5275 -> 5308 (the file grows 5267 -> 5308). `library_screen.py` is unchanged by this round (a one-line swap at the call site).
