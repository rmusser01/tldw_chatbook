---
id: TASK-41601
title: "Folder files mode leaves focus on the hidden Database Notes filter, and a Notes rail entry aims at the hidden Database rows"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-18 09:13'
created_date: 2026-11-17 19:40
labels:
- follow-up
- library
- focus
- ui
priority: medium
references:
- tldw_chatbook/UI/Library_Modules/library_notes_controller.py
- tldw_chatbook/UI/Screens/library_screen.py
- Tests/UI/test_library_shell.py
dependencies:
- TASK-41129
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the libfocus stream (TASK-41129 fix round 4, 2026-11-17). It happens on base cc6006fcf5 and on head.

A probe of Notes in Folder files mode found two things:
- the Database filter and rows stay mounted but hidden, out of `screen.focus_chain`;
- after the switch to Folder files, focus is left on that hidden filter, and a Notes rail entry aims at the hidden Database rows.

**Code** (read at 212f015a91):
- **The switch.** `_show_library_file_notes` (`library_notes_controller.py:5523`) clears focus through `_evacuate_library_notes_authority_focus("database")` (`:1824`). It then hides the Database shell and calls `_restore_library_notes_authority_focus("files")` (`:5598`). That call focuses nothing when `#file-notes-search` is not yet in the focus chain (`:1836-1856`).
- **No reachability check on the Notes restore.** Neither `_library_notes_fallback_focus_target` (`:2093-2094`, the filter) nor `_restore_library_notes_focus_identity` (`:2202`) checks the focus chain. A Notes restore can therefore land on the hidden filter. The probe did not name which call landed it there.
- **The rail entry.** `_focus_library_list_entry` (`library_screen.py:11503`) maps the Notes rail row to the `library-notes-row` class (`:11513`), and the query finds the hidden Database rows (`:11531`). It then calls `set_focus(rows[0])` (`:11627`, `:11643`) with no focus-chain check. Only the empty-list fallback checks (`:11580-11585`, added by TASK-41129). TASK-41129's fix round 5 removed a Database-only gate on these rules, because it changed nothing in the cases then pinned.

**Related.** TASK-41504 covers code-driven and rail entries that land on a grip or the rail row in Database mode.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After Notes switches to Folder files, focus is on a visible Folder files control or on nothing, never on a hidden Database Notes control
- [x] #2 A Notes rail entry in Folder files mode never focuses a hidden Database row or filter; it lands on the Folder files entry target or a documented empty-state control
- [x] #3 Pilot pins drive both journeys and assert the focused widget id and that it is in `screen.focus_chain`; red on the code before the fix
<!-- AC:END -->

## Implementation Plan

1. A Database Notes canvas sync does not move focus while Folder files owns the route (`_restore_library_notes_after_targeted_sync`).
2. The Folder files entry target is the remembered reachable control, else `#file-notes-search` with a linked folder, else **Choose folder…** (`#file-notes-choose-root`) with none; only a control in `screen.focus_chain` counts.
3. The switch retries the Folder files target once behind its layout sync when the first attempt finds it unreachable. (A Folder files branch in the list entry channel and an entry arm in the switch were tried and removed: neither changed any outcome.)
4. Pilot pins for the switch and for a Notes rail entry in Folder files mode, asserting the focused id and `screen.focus_chain`; red on the base.
5. Folder files guide page; covering tests, ratchets, preflight.

## Implementation Notes

**Reproduced on the base** (`67583dfa72`, 170x48), with a `set_focus` trace.
- **The switch.** `_show_library_file_notes` cleared focus, and its Folder files restore found `#file-notes-search` unreachable. A Database Notes canvas sync's default restore then put focus on the hidden `#library-notes-filter`, 6 of 6 runs.
- **A Notes rail entry in Folder files mode.** It never reached the Database rows: the whole-screen recompose does not compose the Database shell in Folder files mode. It ended instead on `#file-notes-search` with its pane hidden. With no folder linked, `_sync_body_panes` hides the Files pane, and the compose path's restore ran before that.

**Fix.**
- `_restore_library_notes_after_targeted_sync` moves no focus while Folder files owns Notes (`_file_notes_active()`), so a sync of the hidden Database canvas leaves focus alone.
- The Folder files entry target (`_restore_library_notes_authority_focus("files")`), used by the switch and by the compose path, is the remembered reachable control first. Then it is `#file-notes-search` with a folder linked, or **Choose folder…** (`#file-notes-choose-root`) with none. It still lands only on a control in `screen.focus_chain`.
- With a folder linked, the switch's first attempt finds the Files pane not shown yet, so the switch retries once behind its own layout sync.

**Removed during the work, because no test or probe could tell them apart.**
- A Folder files branch in `_focus_library_list_entry`. Even a code-driven switch inside the entry window never let the entry aim at a hidden row (3 of 3 probes, with and without the branch).
- An entry-channel arm in the switch. The direct restore always landed first.

**Tests** (`Tests/UI/test_library_notes_move_never_false_conflicts.py`, PR-laned, no census edit).
- `test_folder_files_focus_lands_on_a_visible_files_control[switch|rail-entry × no-folder|linked]` drives a real click on **Folder files**. `rail-entry` then clicks the visible Notes rail row.
- It asserts the focused id (`file-notes-choose-root`, or `file-notes-search` with a folder linked) and `screen.focus_chain` membership after the entry window.
- It records every `set_focus` write into the hidden Database shell and asserts there are none.

**Red evidence** (temporary edits, all reverted).
- **Base source:** three cases failed. `switch` failed with both roots on the hidden `library-notes-filter`. `rail-entry-no-folder` failed on the hidden `file-notes-search`. `rail-entry-linked` passed: the compose path's restore keeps it.
- **Head without that compose-path restore:** both `rail-entry` cases failed (focus None).
- **Head without the switch's retry:** `switch-linked` failed (focus None).
- **Head without the Folder files guard:** at first all four cases passed, because a later restore moved focus off the hidden filter. The write recorder was added for that reason; with it, the same mutant fails 4 of 4 (rechecked on the final source). Lesson added to `lessons-testing-evidence.md`.

**Runs.** Final source, the four cases x 10 with 16 CPU burners at `-n 6`: all passed (load 14-43), in the 170-node run noted in TASK-41504. The Folder files UI files in the covering run (`test_library_file_notes_workspace.py`, `test_library_notes_wave_file_notes.py`, `test_library_notes_files_sync_journey.py`, `test_library_notes_w5_folder_files_layout.py`, `test_library_crit9_notes.py`) showed no failure that does not also fail on the base.

**Files.** `UI/Library_Modules/library_notes_controller.py`, `Tests/UI/test_library_notes_move_never_false_conflicts.py`, `Docs/User_Guide/library/file-notes.md`, `backlog/docs/lessons-testing-evidence.md`.

**Docs.** `file-notes.md`: where focus lands on entering Folder files, and the keyboard linking step that no longer needs a Tab.
