---
id: TASK-41128
title: "Owner decision needed: Escape from Library Notes Create or Add from files lands on the Notes filter, not on the control that opened it"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-16 21:22'
created_date: 2026-11-15 02:20
labels:
- follow-up
- library
- notes
- owner-decision
- ux
priority: low
references:
- tldw_chatbook/UI/Library_Modules/library_notes_controller.py
- Tests/UI/test_library_shell.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** A keyboard user who opens Create (or Add from files) and presses Escape expects to return to where they started; instead focus lands in the filter field, where the next letter types into the filter. TASK-40294 found the behaviour is deliberate and re-pinned its test to it, so whether it is right is a product question this task asks; it does not decide it.

**Evidence.** `library_notes_controller.py:5689-5772` (`_try_switch_retained_library_notes_route`) restores the navigator with `semantic_role="create-template:blank" if entering_create else "filter"` (`:5767`). TASK-40294's notes: both Escape steps in `test_library_note_footer_covers_navigator_create_sync_and_exit` "restore the navigator's `filter` role on purpose", and a step walk showed focus in `#library-notes-filter` at both, with the footer reading `esc rail | typing in field`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner's decision on where Escape from Create and from Add from files lands is recorded verbatim in the notes
- [x] #2 The behaviour and the footer follow the decision, the Library Notes guide page says where focus lands, and the test pins it
<!-- AC:END -->

## Implementation Plan

1. Record the owner's answer verbatim in the notes.
2. Find the page's own blank-center state and its focus landing: the work pane's `list` mode ("Select a note to edit it here.", `#library-note-work-empty`), entered through the shared list entry-focus channel (`_arm_library_list_entry_focus` -> first `.library-notes-row`; the filter only for an empty list, task-31946).
3. Point the Create exit (`_try_switch_retained_library_notes_route`, leaving Create) and the Add from files exits (the chooser's `_exit_library_notes_lasting_sync`, Import once's Escape and ‹ Notes) at that channel instead of the filter. Leave Manage sync folders, Trash and the editor exits alone: the decision does not cover them.
4. Re-pin `test_library_note_footer_covers_navigator_create_sync_and_exit` (compact) to the navigator tier with focus on a note row, and add a wide pin that the center shows its blank state; show both red on the old landing.
5. Update the Library Notes guide page; run covering tests, ratchets, ruff, preflight.

## Implementation Notes

**Owner decision (verbatim, 2026-11-15):** "Library notes page/blank center"

Read as: Escape from Create or from Add from files returns to the Library Notes page with its center blank (no note, no Create template open) and focus not in the filter.

- **Blank center.** The work pane's `list` mode, "Select a note to edit it here." (`#library-note-work-empty`); below 64 columns the same state is the list alone on the stage. No second existing state fits "the Library notes page": the only other blank surface, the Library landing, is not the Notes page. So nothing had to be picked between two.
- **Focus.** Where Notes entry puts it on that page: the shared list entry-focus channel (`_arm_library_list_entry_focus` -> `_focus_library_list_entry`) lands on the first note row (`.library-notes-row`, `#library-notes-tree-note-1` in the tests). With no notes the channel's own fallback is the filter (task-31946), so an empty list still lands there; see the report's follow-up.
- **Exits changed (Escape and its ‹ Notes twin share each one):** Create (`_try_switch_retained_library_notes_route`, leaving Create); the Add from files chooser and Keep a folder synced canvas (`_exit_library_notes_lasting_sync` for the `lasting_add` view); Import once (its Escape branch and `handle_library_notes_import_back`). Manage sync folders (`lasting_roots`) keeps the filter: it was not part of the decision.
- **Footer.** No footer code changed: with focus on a note row the Notes tier shows ("n new | / find | g folder | esc rail" compact), replacing "esc rail | typing in field". Pinned in the compact node.
- **Tests.** `test_library_note_footer_covers_navigator_create_sync_and_exit` (60x20) re-pinned from TASK-40294's filter landing to the note row and navigator footer. New `test_escape_from_create_and_add_from_files_lands_on_the_blank_notes_page` (170x48, production CSS): Escape from Create, from the chooser and from Import once, and Import once's ‹ Notes, each land on the first note row with `#library-note-work-empty` shown.
- **Red/green.** With the controller change reverted both nodes fail ("focused=LibraryRailSearchInput(id='library-notes-filter')"); each of the four exits reverted alone fails the wide node (4 of 4 mutants red, load 46-58); both pass with the change.
- **Doc claims checked (temporary test, deleted).** Create's ‹ Notes and the chooser's ‹ Notes, clicked, also land on `#library-notes-tree-note-1`. With no notes, Escape from Create lands in `#library-notes-filter`.
- **Docs.** `Docs/User_Guide/library/notes.md`: the New note view and Add from files sections say where Escape / ‹ Notes leave focus.
- **Files.** `UI/Library_Modules/library_notes_controller.py`, `Tests/UI/test_library_shell.py`, `Tests/Architecture/test_library_modules_size_ratchet.py` (7442 -> 7454), `Docs/User_Guide/library/notes.md`.
- **Lane.** Both nodes live in `Tests/UI/test_library_shell.py`, beside the node TASK-40294 re-pinned; that file runs on the push-to-dev `library-shell-dev` job, not on a PR lane, so no census entry was added.
- **Covering run (shared with TASK-41129, final source).** 23 Library files that reach the changed seams (route swap residency and storm, rail switch, focus-never-lands, Media return settlement, Add from files, lasting sync, Import once, crit8 keyboard and shell, w3/w5, onboarding, both Notes widget files) plus every Notes node of `test_library_shell.py`, `-n 4`, load 85-110: 482 passed, 197 failed. All 197 re-run on the base source with the same command (load ~70): 181 fail there too, 162 of them the local `RecoveryRequired: raw_source_selection_changed` wall and the rest assertion failures (including the deselected `test_library_shell_note_save_then_back_refreshes_list_title_and_age`). The 16 that passed on base passed on head when re-run (16 of 16, load 53-73): load flakes from the first run, not regressions. `Tests/Architecture` ratchets and Notes wiring: 90 passed (serial). Ruff gate: OK.
- **Stability.** The three nodes (six cases) passed 3 of 3 runs at `-n 4` (load 33-39). Serial times with the UI-lane flags (load 20-35): footer node 9.4 s, wide node 12.1 s, the four Media -> Notes cases 6.0-7.0 s each (call).
