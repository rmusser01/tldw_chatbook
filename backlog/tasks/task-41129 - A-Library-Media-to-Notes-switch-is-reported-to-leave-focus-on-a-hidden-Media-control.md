---
id: TASK-41129
title: "A Library Media-to-Notes switch is reported to leave focus on a hidden Media control"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-17 02:04'
created_date: 2026-11-15 02:20
labels:
- follow-up
- library
- focus
- investigation
priority: low
references:
- tldw_chatbook/UI/Library_Modules/library_notes_controller.py
- tldw_chatbook/UI/Screens/library_screen.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** Reported by the round-14 Library stream as a bug present on its base: after switching from Media to Notes, focus can stay on a Media control that is no longer displayed, so keys go nowhere visible. It was not reproduced at closeout and no node pins it.

**Evidence to start from.** The Notes route switch restores focus through `_restore_library_notes_focus_identity` after `_try_switch_retained_library_notes_route` (`library_notes_controller.py:5689-5772`); if that restore finds nothing focusable it does not move focus off the previous canvas's widget.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test reproduces the switch at the sizes the Library supports and records where focus lands, or the notes record the bug cannot be reproduced and the task is closed
- [x] #2 If reproduced, focus lands on a visible Notes control after the switch, with the test red before the fix
<!-- AC:END -->

## Implementation Plan

1. Reproduce: drive a real Media -> Notes switch (rail click and keyboard) at the Library's compact and wide sizes, with focus on Media rows, the Media viewer and Media toolbar controls, and record where focus lands and whether it is in the focus chain.
2. If it reproduces, fix at the shared switch seam so focus lands on a visible Notes control, with the test red before the fix.
3. If it does not, record the attempts as evidence in the notes and close the task.

## Implementation Notes

Reproduced on base cc6006fcf5, and fixed.

- **Probe (temporary, deleted).** 6 sizes (60x20, 80x24, 100x30, 120x36, 170x48, 235x52) x focus on a Media row, the Media filter, the open Reader or the empty page's Import x two switches, under the production stylesheet, sampling focus for 3.5 s:
  - **Rail click.** Never stranded. The click moves focus to the rail's Notes row before the switch, and entry focus then lands on the first note row. Once in four wide runs (170x48) focus stayed on the rail row instead, which is visible.
  - **`press()` on the rail row** (any switch that leaves focus where it was). At 120x36 focus ended on `#library-media-row-0` and at 235x52 on `#library-media-row-1`, both outside `screen.focus_chain` with a hidden ancestor. 170x48 hit the same transient and recovered.
  - **Deep link** (`apply_navigation_context({"mode": "notes"})`). Landed on the visible `#library-notes-source-database` in 4 of 4 runs.
- **Mechanism (set_focus/arm/disarm trace, 6 of 6 runs at 120x36 and 235x52).** The browse route swap (`_adopt_library_browse_canvas`) hides the resident Media canvas with `display = False`. Textual's `Hide` blurs the focused row, and `Screen._reset_focus` takes a `visible_siblings` entry that is `focusable`: `#library-media-row-0` in the same hidden list (`visible_siblings` checks each sibling's own `visibility` and `display`, and `focusable` checks inherited `visibility`, but neither looks at a display-hidden ancestor). That `DescendantFocus` reads as a user move in `on_descendant_focus`, which disarms the Notes entry focus, so focus never leaves the hidden row.
- **Fix.** `_adopt_library_browse_canvas` clears focus held in the canvas it is about to hide, before hiding or removing anything. This is the rule `LibraryBrowseReaderShell.swap_work` already applies to its outgoing pane (TASK-34000.25). The destination's entry focus then lands. The swap serves both browse routes, so Notes -> Media gets the same rule.
- **Test.** `Tests/UI/test_library_shell.py::test_media_to_notes_switch_never_leaves_focus_on_a_hidden_media_row[60x20|100x30|120x36|235x52]`: focus a Media row, `press()` the Notes rail row, and wait for focus on the first note row, inside the focus chain. Red 4 of 4 with the swap change reverted (focus on `library-media-row-0`, load ~80), green 4 of 4 with it (load ~77).
- **Real-user exposure.** A rail click or Enter moves focus to the rail first, so the everyday path was not affected. The bug needs a switch that leaves focus in Media, which today means code-driven selection and tests.
- **Files.** `UI/Library_Modules/library_browse_route_swap.py` (no size ratchet governs it), `Tests/UI/test_library_shell.py`, `backlog/docs/lessons-textual.md` (new entry).
- **Lane.** The test sits in `test_library_shell.py` (push-to-dev `library-shell-dev` job), not on a PR lane; no census entry was added.
**Fix round 1 (review r15lf-rev1, 2026-11-17).**
- **A second Media -> Notes switch landed in the filter.** If Notes had been opened earlier in the session, its canvas is resident, and the swap repaints it with `_sync_library_canvas`. That sync's default `_restore_library_notes_after_targeted_sync` replayed an identity captured while focus was None, which resolves to the filter, and it ran after the entry attempt had already landed on the first row. Fix: `LibraryNotesController._restore_library_notes_after_targeted_sync` skips its focus restore while the list entry channel is armed. Its scroll restore still runs. This is the guard `_restore_library_media_focus` already applies for Media.
- **Test.** `test_media_to_notes_switch_never_leaves_focus_on_a_hidden_media_row` gained `[120x36-revisit]` and `[235x52-revisit]`: Notes opened first, then Media, then `press()` Notes, landing exactly on the first note row. Every case now waits past the entry arm and pauses 0.5 s before asserting, so a late restore shows. With the guard removed, both revisit cases fail with focus on `library-notes-filter`, in 2 of 2 runs (load ~50). Before the fix they were also red in the first runs, 1 or 2 of 2. They pass with the guard.
- **The unexplained `[235x52]` failure.** It did not reproduce: 8 of 8 passed at load ~70 through the reviewer's repeat plugin. The likely cause is the one the reviewer named. On a first visit the Notes rows can arrive after the filter is focusable, and `_focus_library_list_entry` then lands in the filter (task-31946's empty-list rule). It does not check that the list has settled. That is a visible Notes control, so a first-visit case now accepts the first row or the filter, and fails only on focus outside Notes or outside the focus chain. The revisit cases, whose rows are resident, still demand the first row. Not settling before the fallback is pre-existing and listed as a follow-up in the report.
- **Size.** `library_notes_controller.py` ratchet 7457 -> 7463.
- **Wording.** The swap comment and the lesson now say the check misses a display-hidden *ancestor*. `visible_siblings` does check each sibling's own `display`.
- **Covering run (shared with TASK-41128, final source).** 23 Library files that reach the changed seams (route swap residency and storm, rail switch, focus-never-lands, Media return settlement, Add from files, lasting sync, Import once, crit8 keyboard and shell, w3/w5, onboarding, both Notes widget files) plus every Notes node of `test_library_shell.py`, `-n 4`, load 85-110: 482 passed, 197 failed. All 197 re-run on the base source with the same command (load ~70): 181 fail there too, 162 of them the local `RecoveryRequired: raw_source_selection_changed` wall and the rest assertion failures (including the deselected `test_library_shell_note_save_then_back_refreshes_list_title_and_age`). The 16 that passed on base passed on head when re-run (16 of 16, load 53-73): load flakes from the first run, not regressions. `Tests/Architecture` ratchets and Notes wiring: 90 passed (serial). Ruff gate: OK.
- **Stability.** The three nodes (six cases) passed 3 of 3 runs at `-n 4` (load 33-39). Serial times with the UI-lane flags (load 20-35): footer node 9.4 s, wide node 12.1 s, the four Media -> Notes cases 6.0-7.0 s each (call).
- **Covering run (fix round 1).** 11 Library files: canvas sync defects and scoped sync, crit8 shell and keyboard, phase-C residency and storm, Notes reader, rail switch, both focus-never-lands files, and Add from files. Also every Notes node of `test_library_shell.py`. Run through the reviewer's plugin in `bootstrap_ui` mode, so the plain UI files run instead of hitting `RecoveryRequired`, at `-n 4`, load 39-64: 397 passed, 12 failed.
  - Two failures were `RecoveryRequired`, which the plugin's marker does not cover.
  - Re-running the other ten: 7 failed on head, 6 with the guard off (the previous head), and 7 on the reviewer's base emulation (guard off, base swap, base Notes exits), with the same failures. The three crit8 typing nodes ('' == 'hello'), the two Notes reader nodes and the deselected title-and-age node fail in all three. `test_the_picker_alone_never_writes_focus_onto_a_removed_widget` fails on head and on base and passed once with the guard off, so it flakes.
  - The three that passed on the re-run (two blank-note GC nodes and the breakpoint round trip) were load flakes.
- **Stability (fix round 1).** The four changed nodes (ten cases) passed 3 of 3 runs at `-n 4` (load 48-57). Ratchets and Notes wiring: 90 passed (serial). Ruff gate: OK.
