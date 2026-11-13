---
id: TASK-40297
title: "The Library Export quality strip leaves the ‹ Library bar enabled while the strip guards the canvas"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-13 07:20'
created_date: 2026-11-12 19:36
labels:
- follow-up
- library
- export
priority: low
references:
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/library_export_controller.py
- Tests/UI/test_library_shell.py
- scripts/library_shell_push_deselect.txt
dependencies:
- TASK-39152
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down: TASK-39152 classified the node as a product defect and left it red. Checked at the round-13 stack top 8b633d0101, 2026-11-12: the nodes below fail when run together (run.sh, `-n 4`, load ~13), with the failure quoted.

**What the user sees.** While the Export quality strip is open and guarding the canvas, the emergency "‹ Library" bar stays enabled. It offers a way out that the strip's own guard should be deciding.

**Cause, from TASK-39152's notes.** Since 43849abe13 (TASK-21243) the strip opens canvas-scoped, and nothing re-syncs the bar's guard presentation after the open. A temporary `_sync_library_emergency_guard_presentation()` call after the open (`UI/Screens/library_screen.py:9118`) made the node pass.

**Node.** `Tests/UI/test_library_shell.py::test_library_emergency_real_guards_keep_specific_action_authoritative[cancel]`: `LibraryEmergencyReturn(id='library-emergency-return').disabled` is False.

The `test_library_shell.py` nodes are deselected from the `library-shell-dev` push job (`scripts/library_shell_push_deselect.txt`, TASK-39153); remove each line when its fix lands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 While the Export quality strip guards the canvas, the ‹ Library bar is disabled, and it is enabled again once the strip closes
- [x] #2 The node passes 3 of 3 at `-n 4`, shown red before the fix, and its line leaves `scripts/library_shell_push_deselect.txt`
<!-- AC:END -->

## Implementation Plan

1. Call `_sync_library_emergency_guard_presentation()` where the Export quality strip opens or closes (the toggle and a choice pick), as the Escape close already does.
2. Show the node red with the fix reverted; remove its deselect line.

## Implementation Notes

Premise held on base 26b2e677dd (the node failed with `LibraryEmergencyReturn.disabled` False, load ~28).

- `LibraryExportController.handle_library_export_quality` (open and close) and `handle_library_export_quality_choice` now call `_sync_library_emergency_guard_presentation()` right after flipping `quality_choices_visible`, as the Escape close (`_close_open_library_choice_strip`) already did. The eligibility projection already counted the open strip as a guard; nothing repainted the bar after the canvas-scoped open.
- Scope note: the media/prompts/skills strips have the same shape, but their routes use reader shells, where the emergency (canvas-only) stage never applies, so only Export can show the bar with a strip open.
- Red/green: with the two calls removed, `test_library_emergency_real_guards_keep_specific_action_authoritative[cancel]` fails on `bar.disabled is True` (load ~69); with them, both parameters pass (3 of 3 at `-n 4`, load ~41: the eleven stream nodes passed 11/11 in each of three runs).
- Files: `UI/Library_Modules/library_export_controller.py`, `Tests/Architecture/test_library_modules_size_ratchet.py` (row 1501 -> 1504), `scripts/library_shell_push_deselect.txt`, `Docs/User_Guide/library/import-and-export.md`.

Covering run (shared by the stream's five tasks, final tree): 58 Library UI files that reference the changed seams (`test_library_shell.py` whole, side-by-side, canvas-sync defects, choice strips, multiselect, honesty, workbench focus, screen navigation and 50 more), `-n 4`, load 42-66: 2023 passed, 863 failed. Every failing function was re-run on the head tree and on the base source (26b2e677dd, same tests): of those 871 nodes 850 fail on head and 845 on base, almost all the local RecoveryRequired wall for plain Tests/UI files plus the deselected nodes owned by TASK-32503/34000.60/40296/40299. Head-only reds re-checked: `test_focus_library_list_entry_falls_back_to_first_row_outside_active_selection` (fake rows without `ancestors`; fixed in TASK-40293), `test_any_non_markdown_item_with_content_says_why_rendered_is_absent[video]` (3 of 3 green on head when re-run alone), `test_literal_history_restore_keeps_kind_definition_and_guards` (red 1-3 of 3 params on base too), and four nodes that passed on re-run. `Tests/Architecture` Library wiring files: 49 passed (serial).

**Fix round 1 (review r14lf-rev1).** The pick close at `library_export_controller.py` had no test (`[cancel]` closes with Escape). The node now also has `[pick]` (picks a choice) and `[toggle]` (presses the opener a second time). Each closes the strip its own way, then asserts `bar.disabled is False` and `esc == "rail"`. With only the pick handler's sync removed, `[pick]` fails on `bar.disabled` still True while `[toggle]` and `[cancel]` pass. With the toggle handler's sync removed, all three open-strip parameters fail at the open step. All four parameters pass on the fixed tree.
