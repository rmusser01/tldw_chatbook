---
id: TASK-40294
title: "The compact Library Notes footer keeps the rail tier for 2 s or more after Notes entry has put focus on a note row"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-13 05:40'
created_date: 2026-11-12 19:36
labels:
- follow-up
- library
- notes
- footer
priority: low
references:
- tldw_chatbook/UI/Library_Modules/library_notes_controller.py
- tldw_chatbook/UI/Screens/library_screen.py
- Tests/UI/test_library_shell.py
- scripts/library_shell_push_deselect.txt
dependencies:
- TASK-39152
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down: TASK-39152 classified the node as a product defect and left it red. Checked at the round-13 stack top 8b633d0101, 2026-11-12: the nodes below fail when run together (run.sh, `-n 4`, load ~13), with the failure quoted.

**What the user sees.** On compact Notes entry the system's entry focus lands on a note row, but the footer still shows the rail tier for 2 s or more. The navigator keys `n new` and `g folder` already work, but the footer does not advertise them.

**Node.** `Tests/UI/test_library_shell.py::test_library_note_footer_covers_navigator_create_sync_and_exit`: "Footer did not settle to 'n new | / find | g folder | esc rail'; got '/ focus search | F6 next pane | esc focus rail'."

The `test_library_shell.py` nodes are deselected from the `library-shell-dev` push job (`scripts/library_shell_push_deselect.txt`, TASK-39153); remove each line when its fix lands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After compact Notes entry puts focus on a note row, the footer shows the navigator tier within the test's settle window
- [x] #2 The node passes 3 of 3 at `-n 4`, shown red before the fix, and its line leaves `scripts/library_shell_push_deselect.txt`
<!-- AC:END -->

## Implementation Plan

1. Trace what writes the footer after compact Notes entry (footer writes and registrations with stacks).
2. Fix at the seam: `_register_footer_shortcuts` derives its set through the Notes dispatcher (`_library_notes_footer_shortcuts`), which falls back to the generic set off Notes, so a pane-visibility re-registration cannot overwrite the Notes tier.
3. Walk the rest of the node; re-pin any step that drifted for a reason outside this defect, with the reason in a comment.
4. Show the node red with the fix reverted; remove its deselect line.

## Implementation Notes

Premise held on base 26b2e677dd. A footer-write probe showed the cause: `_library_pane_visibility_changed` calls `_register_footer_shortcuts()`, which registered the generic `_library_footer_shortcuts_for_current_state()` (`LIBRARY_LIST_SHORTCUTS`) over the Notes tier, and `_apply_library_notes_footer_context` dedups on the registration, so nothing put the Notes tier back until another state change.

- Fixed at the seam: `_register_footer_shortcuts` now takes its set from `_library_notes_footer_shortcuts()`, which already falls back to the generic set whenever the Notes workflow is not active. Every caller (compose, pane visibility, Media viewer focus, guard repaint) now registers what the Notes dispatcher would; the same overwrite was what Qodo #2359 fixed for one caller.
- With the first step green, the node's two later "back to the navigator" steps (Escape from Create and from Add-from-files) had drifted: both transitions restore the navigator's `filter` role on purpose (`_try_switch_retained_library_notes_route`), and since TASK-34000.40 (N-31, 2026-10-26) a focused text field shows `esc rail | typing in field`. Those two pins (last edited 2026-09-15, before N-31) were re-pinned to that tier with a comment; a step walk showed focus in `#library-notes-filter` at both.
- Red/green: with `_register_footer_shortcuts` back on the generic set, the node fails at its first step ("got '/ focus search | F6 next pane | esc focus rail'", load ~65); with the fix it passes (3 of 3 at `-n 4`, load ~41: the eleven stream nodes passed 11/11 in each of three runs).
- Files: `UI/Screens/library_screen.py` (+2, ratchet 38183 -> 38185), `Tests/UI/test_library_shell.py`, `Tests/Architecture/test_screen_size_ratchet.py`, `scripts/library_shell_push_deselect.txt`.

Covering run (shared by the stream's five tasks, final tree): 58 Library UI files that reference the changed seams (`test_library_shell.py` whole, side-by-side, canvas-sync defects, choice strips, multiselect, honesty, workbench focus, screen navigation and 50 more), `-n 4`, load 42-66: 2023 passed, 863 failed. Every failing function was re-run on the head tree and on the base source (26b2e677dd, same tests): of those 871 nodes 850 fail on head and 845 on base, almost all the local RecoveryRequired wall for plain Tests/UI files plus the deselected nodes owned by TASK-32503/34000.60/40296/40299. Head-only reds re-checked: `test_focus_library_list_entry_falls_back_to_first_row_outside_active_selection` (fake rows without `ancestors`; fixed in TASK-40293), `test_any_non_markdown_item_with_content_says_why_rendered_is_absent[video]` (3 of 3 green on head when re-run alone), `test_literal_history_restore_keeps_kind_definition_and_guards` (red 1-3 of 3 params on base too), and four nodes that passed on re-run. `Tests/Architecture` Library wiring files: 49 passed (serial).
