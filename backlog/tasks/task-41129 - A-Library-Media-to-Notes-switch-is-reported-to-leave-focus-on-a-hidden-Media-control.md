---
id: TASK-41129
title: "A Library Media-to-Notes switch is reported to leave focus on a hidden Media control"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-16 21:22'
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
- **Mechanism (set_focus/arm/disarm trace, 6 of 6 runs at 120x36 and 235x52).** The browse route swap (`_adopt_library_browse_canvas`) hides the resident Media canvas with `display = False`. Textual's `Hide` blurs the focused row, and `Screen._reset_focus` takes a `visible_siblings` entry that is `focusable`: `#library-media-row-0` in the same hidden list (both checks read `visibility`, not `display`). That `DescendantFocus` reads as a user move in `on_descendant_focus`, which disarms the Notes entry focus, so focus never leaves the hidden row.
- **Fix.** `_adopt_library_browse_canvas` clears focus held in the canvas it is about to hide, before hiding or removing anything. This is the rule `LibraryBrowseReaderShell.swap_work` already applies to its outgoing pane (TASK-34000.25). The destination's entry focus then lands. The swap serves both browse routes, so Notes -> Media gets the same rule.
- **Test.** `Tests/UI/test_library_shell.py::test_media_to_notes_switch_never_leaves_focus_on_a_hidden_media_row[60x20|100x30|120x36|235x52]`: focus a Media row, `press()` the Notes rail row, and wait for focus on the first note row, inside the focus chain. Red 4 of 4 with the swap change reverted (focus on `library-media-row-0`, load ~80), green 4 of 4 with it (load ~77).
- **Real-user exposure.** A rail click or Enter moves focus to the rail first, so the everyday path was not affected. The bug needs a switch that leaves focus in Media, which today means code-driven selection and tests.
- **Files.** `UI/Library_Modules/library_browse_route_swap.py` (no size ratchet governs it), `Tests/UI/test_library_shell.py`, `backlog/docs/lessons-textual.md` (new entry).
- **Lane.** The test sits in `test_library_shell.py` (push-to-dev `library-shell-dev` job), not on a PR lane; no census entry was added.
- **Covering run (shared with TASK-41129, final source).** 23 Library files that reach the changed seams (route swap residency and storm, rail switch, focus-never-lands, Media return settlement, Add from files, lasting sync, Import once, crit8 keyboard and shell, w3/w5, onboarding, both Notes widget files) plus every Notes node of `test_library_shell.py`, `-n 4`, load 85-110: 482 passed, 197 failed. All 197 re-run on the base source with the same command (load ~70): 181 fail there too, 162 of them the local `RecoveryRequired: raw_source_selection_changed` wall and the rest assertion failures (including the deselected `test_library_shell_note_save_then_back_refreshes_list_title_and_age`). The 16 that passed on base passed on head when re-run (16 of 16, load 53-73): load flakes from the first run, not regressions. `Tests/Architecture` ratchets and Notes wiring: 90 passed (serial). Ruff gate: OK.
- **Stability.** The three nodes (six cases) passed 3 of 3 runs at `-n 4` (load 33-39). Serial times with the UI-lane flags (load 20-35): footer node 9.4 s, wide node 12.1 s, the four Media -> Notes cases 6.0-7.0 s each (call).
