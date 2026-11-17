---
id: TASK-41128
title: "Owner decision needed: Escape from Library Notes Create or Add from files lands on the Notes filter, not on the control that opened it"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-17 08:33'
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
- **Focus.** Where Notes entry puts it on that page: the shared list entry-focus channel (`_arm_library_list_entry_focus` -> `_focus_library_list_entry`) lands on the first note row (`.library-notes-row`, `#library-notes-tree-note-1` in the tests). With no notes the channel's own fallback is the filter (task-31946); the follow-up round below lands those exits on New instead.
- **Exits changed (Escape and its ‹ Notes twin share each one):** Create (`_try_switch_retained_library_notes_route`, leaving Create); the Add from files chooser and Keep a folder synced canvas (`_exit_library_notes_lasting_sync` for the `lasting_add` view); Import once (its Escape branch and `handle_library_notes_import_back`). In the first round Manage sync folders (`lasting_roots`) kept the filter; the follow-up round below moved it.
- **Footer.** No footer code changed: with focus on a note row the Notes tier shows ("n new | / find | g folder | esc rail" compact), replacing "esc rail | typing in field". Pinned in the compact node.
- **Tests.** `test_library_note_footer_covers_navigator_create_sync_and_exit` (60x20) re-pinned from TASK-40294's filter landing to the note row and navigator footer. New `test_escape_from_create_and_add_from_files_lands_on_the_blank_notes_page` (170x48, production CSS): Escape from Create, from the chooser and from Import once, and Import once's ‹ Notes, each land on the first note row with `#library-note-work-empty` shown.
- **Red/green.** With the controller change reverted both nodes fail ("focused=LibraryRailSearchInput(id='library-notes-filter')"); each of the four exits reverted alone fails the wide node (4 of 4 mutants red, load 46-58); both pass with the change.
- **Doc claims checked (temporary test, deleted).** Create's ‹ Notes and the chooser's ‹ Notes, clicked, also land on `#library-notes-tree-note-1`. With no notes, Escape from Create landed in `#library-notes-filter` (first round; New since the follow-up round).
- **Docs.** `Docs/User_Guide/library/notes.md`: the New note view and Add from files sections say where Escape / ‹ Notes leave focus.
- **Files.** `UI/Library_Modules/library_notes_controller.py`, `Tests/UI/test_library_shell.py`, `Tests/Architecture/test_library_modules_size_ratchet.py` (7442 -> 7454), `Docs/User_Guide/library/notes.md`.
- **Lane.** Both nodes live in `Tests/UI/test_library_shell.py`, beside the node TASK-40294 re-pinned; that file runs on the push-to-dev `library-shell-dev` job, not on a PR lane, so no census entry was added.
**Follow-up round (coordinator rulings, 2026-11-16).** These are the coordinator's engineering calls applying the owner's rule; neither was the owner's own answer.
- **Zero notes.** The coordinator ruled that, when one of these exits finds no note row, focus lands on the blank page's own primary control instead of the filter. The ruling covers the exits only: entering an empty Notes list through the rail from anywhere other than Create keeps its existing entry rule (see Fix round 1). The page offers one: the empty list's copy is "No notes yet. Create your first note." (`#library-notes-empty`, a Static) and its create action is the toolbar's **New** (`#library-notes-new`). The blank work pane ("Select a note to edit it here.") has no focusable control. A filter that matches nothing takes the same branch, since it also has no row.
- **Manage sync folders.** The coordinator extended the owner's rule to this flow for consistency: its Escape and ‹ Notes now land like the other Notes sub-flows (first note row, or New with no notes).
- **Code.** One helper, `_land_on_library_notes_page`, is now the `then` of every exit (Create, the Add from files chooser and Keep synced, Import once Escape and ‹ Notes, Manage sync folders). It focuses New when `#library-notes-empty` is showing and New is focusable; otherwise it arms the list entry channel. The rail's Notes row pressed from Create goes through the same Create exit (the code cannot tell it from Escape), so it lands the same way; see Fix round 1.
- **Tests.** The wide node is parametrized `[two-notes|no-notes]`, runs with a one-root fake sync runtime so the toolbar offers Manage sync folders, and adds that flow's Escape and ‹ Notes. `test_lasting_setup_escape_abandons_review_before_restoring_notes_focus` (zero notes, Escape from a Keep synced review) pinned the filter; it now waits for New.
- **Red/green.** Two mutants. With the New branch removed, `[no-notes]` fails at Create and the lasting-setup node fails, both on `library-notes-filter`. With `lasting_roots` sent back to the filter, both parameters fail at "Manage sync folders escape". All pass with the change (load 85-104).
- **Docs and size.** `notes.md`: the zero-notes landing, plus a Manage sync folders sentence. Ratchet 7454 -> 7457.
**Fix round 1 (review r15lf-rev1, 2026-11-17).**
- **Rail Notes row from Create (coordinator ruling).** Leaving Create through the rail's Notes row is a Create exit, which keeps it consistent with the owner's rule. It lands on the first note row, or on New with no note row. Pinned as a step of the wide node (`pilot.click` on the rail row). Rail entry into an empty Notes list from anywhere else keeps its existing entry rule. That is task-31946's filter fallback, but a probe at 170x48 also saw the Library grip from the landing and from Media in some runs, and saw it on the reviewer's base emulation too. It is pre-existing and outside this task; see the report.
- **Footer and center pins.** The wide node now pins the landing footer: "n new note | / find note | g go to folder | esc focus rail" with notes, and "n new note | / find note | esc focus rail" with none. It also asserts that `#library-note-work-empty` has a non-zero region and no hidden ancestor at 170x48.
- **Docs.** In `notes.md` the landing sentence has left the 60-column paragraph and now reads for both widths: "‹ Notes" wide, "‹ Back to list" compact, and the "Select a note…" pane only on a wide terminal. The sentence now names the rail's Notes row, a kept filter that matches nothing, and Keep a folder synced's setup and review. The Manage sync folders bullet names the no-match case too. A probe confirmed the no-match case lands on New (Create Escape and Add from files Escape, wide).
- **Red/green** (temporary edits, all reverted):
  - Rail step: the leaving-Create branch disabled, and the test's Escape-from-Create step dropped, so the rail step is the first exit to run. Both parameters fail at "Create, then the rail's Notes row" on `library-notes-filter`.
  - Footer pin: the landing sent to the filter, with the test's focus checks removed. Both fail on "typing in field | esc focus rail".
  - Center check: the blank pane given width 0. Both fail at "Create".
  - All pass with the change (load 46-55).
- **Covering run (shared with TASK-41129, final source).** 23 Library files that reach the changed seams (route swap residency and storm, rail switch, focus-never-lands, Media return settlement, Add from files, lasting sync, Import once, crit8 keyboard and shell, w3/w5, onboarding, both Notes widget files) plus every Notes node of `test_library_shell.py`, `-n 4`, load 85-110: 482 passed, 197 failed. All 197 re-run on the base source with the same command (load ~70): 181 fail there too, 162 of them the local `RecoveryRequired: raw_source_selection_changed` wall and the rest assertion failures (including the deselected `test_library_shell_note_save_then_back_refreshes_list_title_and_age`). The 16 that passed on base passed on head when re-run (16 of 16, load 53-73): load flakes from the first run, not regressions. `Tests/Architecture` ratchets and Notes wiring: 90 passed (serial). Ruff gate: OK.
- **Covering re-run (follow-up round, final source).** Same 23 files plus the Notes nodes of `test_library_shell.py`, `-n 4`, load 75-127: 482 passed, 198 failed. Of the failures, 162 are the local `RecoveryRequired` wall. The other 36 nodes (33 ids) were re-run on head and then on the base source (cc6006fcf5 controller and swap), same command: 23 failed on each. Three failed on head only and three on base only. Of the head-only three, two passed 3 of 3 on head re-runs. The third, `test_library_shell_blank_note_autosaved_then_emptied_still_gcs_on_back`, failed 2 of 3 serially on head and 2 of 3 on base (load 80-110). It is a 3 s wait for a GC under load, not this change. Architecture ratchets and wiring: 90 passed (serial). Ruff gate: OK.
- **Stability.** The three nodes (six cases) passed 3 of 3 runs at `-n 4` (load 33-39). Serial times with the UI-lane flags (load 20-35): footer node 9.4 s, wide node 12.1 s, the four Media -> Notes cases 6.0-7.0 s each (call).
- **Covering run (fix round 1).** 11 Library files: canvas sync defects and scoped sync, crit8 shell and keyboard, phase-C residency and storm, Notes reader, rail switch, both focus-never-lands files, and Add from files. Also every Notes node of `test_library_shell.py`. Run through the reviewer's plugin in `bootstrap_ui` mode, so the plain UI files run instead of hitting `RecoveryRequired`, at `-n 4`, load 39-64: 397 passed, 12 failed.
  - Two failures were `RecoveryRequired`, which the plugin's marker does not cover.
  - Re-running the other ten: 7 failed on head, 6 with the guard off (the previous head), and 7 on the reviewer's base emulation (guard off, base swap, base Notes exits), with the same failures. The three crit8 typing nodes ('' == 'hello'), the two Notes reader nodes and the deselected title-and-age node fail in all three. `test_the_picker_alone_never_writes_focus_onto_a_removed_widget` fails on head and on base and passed once with the guard off, so it flakes.
  - The three that passed on the re-run (two blank-note GC nodes and the breakpoint round trip) were load flakes.
- **Stability (fix round 1).** The four changed nodes (ten cases) passed 3 of 3 runs at `-n 4` (load 48-57). Ratchets and Notes wiring: 90 passed (serial). Ruff gate: OK.

**Fix round 2 (2026-11-17).** `_land_on_library_notes_page` now only arms the list entry channel with `empty_target="#library-notes-new"`. The channel lands on the first row, or on New once the list has loaded with no note row, and it waits while rows are still loading (TASK-41129 fix round 2). The zero-notes wide case and the lasting-setup node still pass, so New is still the landing for a settled empty list. Controller ratchet 7463 -> 7459.

**Fix round 3 (review r15lf-rev2, 2026-11-17).** An exit taken while the list reloads no longer parks focus on a pane grip. While the rows load, the entry waits with no grip holding focus and lands on the first row when they arrive. If the window closes first it lands on New, the exits' empty-list target. Pinned by `test_an_exit_while_the_notes_list_reloads_never_parks_on_a_grip`; mechanism and evidence are in TASK-41129's fix round 3.

**Fix round 4 (review r15lf-rev3, 2026-11-17).** A user's click on a pane grip during a waiting exit landing now keeps the grip. Only the canvas sync's own restore keeps a waiting entry armed. The landing rules apply only to the Database list, not Folder files. Details and evidence are in TASK-41129's fix round 4.
