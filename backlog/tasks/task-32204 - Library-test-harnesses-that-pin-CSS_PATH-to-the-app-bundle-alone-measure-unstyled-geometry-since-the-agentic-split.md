---
id: TASK-32204
title: >-
  Library test harnesses that pin CSS_PATH to the app bundle alone measure
  unstyled geometry since the agentic split
status: Done
assignee: []
created_date: '2026-09-10 15:20'
updated_date: '2026-09-11 10:45'
labels:
  - library
  - tests
  - css
  - test-health
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-25812 (PR #2281, 2026-08-31) split the Library, Console and Settings rules out of the boot bundle into lazily loaded screen sheets (`screen_agentic_library.tcss` and siblings), and TASK-24459 did the same for Evals and Scheduling. Production loads those sheets through the owning screen's `CSS_PATH` or `TldwCli._SCREEN_OWNED_ROUTE_CSS`, so the real app is styled correctly. Any test harness that sets `CSS_PATH` to the bundle alone has been rendering Library widgets with no Library rules since that day: `Button` falls back to `width: auto`, `Vertical` to `height: 1fr`, and every width, wrap and scroll assertion in such a test measures an unstyled layout.

Three Notes tests were found red on `dev` for exactly this reason and repaired in PR #2564 (`test_compact_pagers_paint_full_wrapped_copy_at_80x24`, `test_nested_pager_paints_projected_depth_indentation`, `test_long_history_keeps_paging_actions_pinned_with_scroll_cue`): bisected to `b62407e258`, not to any product change; `grep -c library-notes-tree-pager tldw_cli_modular.tcss` returns 0; a live probe showed the pager buttons resolving `width = auto` instead of the sheet's `width: 100%`. `Tests/UI/consolidated_css.py` already documents the trap and exposes `APP_STYLESHEETS` as the correct pin. PR #2564 fixed the two harness files it touched and did not sweep the rest of `Tests/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every test harness under `Tests/` that composes a Library, Console, Settings, Evals or Scheduling widget loads the same stylesheet set the app loads for that screen (`APP_STYLESHEETS` or the owning screen's sheet), and the sweep that found them is recorded in the notes with its command and hit list
- [x] #2 A guard test fails when a harness class sets `CSS_PATH` to the bundle path alone while composing a widget whose rules live in a screen-owned sheet, naming the harness and the sheet
- [x] #3 Any test whose expected geometry changes once it is styled is re-pinned to the styled truth with equal or better assertion strength, and each such re-pin cites this task
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Make the sweep executable: resolve each harness `CSS_PATH` statically (inline
   module-level names, same-module `Class.CSS_PATH`, the real `TldwCli.CSS_PATH`),
   skip harnesses that push the owning screen, flag those whose queried selectors
   only the missing split sheet styles.
2. Flip every hit to `APP_STYLESHEETS`, preserving any screen-sheet bracket.
3. Run every touched file against a detached `origin/dev` worktree and compare
   FAILED name sets; re-pin whatever the styling legitimately changes.
4. Keep the scan as the AC#2 guard test.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: the sweep IS the guard. `scan_bundle_only_harnesses` in
`Tests/UI/test_consolidated_css_harness.py` walks every `Tests/**/*.py`, resolves
each class's `CSS_PATH` (two substitution passes over module-level assignments,
same-module `Class.CSS_PATH`, and the real `TldwCli.CSS_PATH`), skips a harness
that names the screen owning the sheet (Textual loads it there, as production
does), and flags one that queries a selector present ONLY in the missing sheet.
Docstrings are excluded from the selector scan -- this repo discusses selectors in
prose constantly, and an early version flagged four `test_library_shell.py`
harnesses off comment text alone.

Sweep command and hit list (recorded per AC#1):

    .venv/bin/python -c "from Tests.UI.test_consolidated_css_harness import \
        scan_bundle_only_harnesses; print(len(scan_bundle_only_harnesses()))"

41 harnesses across 30 files: 25 Console-sheet (test_console_assistant_turn x2,
character_context_geometry, chip_strip_overflow, composer_collapse x2,
composer_cursor, composer_overflow, narrow_layout, prompts_modal,
rail_width_budget, run_inspector, scope_picker_modal, settings_geometry,
setup_card_fit x2, tab_strip_budget x2, thinking_disclosures,
transcript_markdown_widget, turn_file_card x4, workspace_files_integration),
7 scheduling (destination_shells x2, schedules_automations_tab,
schedules_new_button, schedules_workbench x4 -- 8 rows), 5 settings
(settings_agents_category, settings_narrow_layout, settings_web_search,
speech_tts_settings_ownership_closeout), 3 library
(library_adaptive_reader_shell `_ProbeApp`, library_ingest_structural
`_CssTrueCanvasHost`, product_maturity_gate1 `ConsoleHarness`), 1 watchlists
(kept_briefings_modal). All 41 now pin `APP_STYLESHEETS`; the scan returns 0.

Evidence the pin matters (AC#1): with the sheets loaded, five tests that are RED
on dev pass -- `test_inspector_group_heading_shares_a_left_edge_with_its_rows`,
`test_proprietary_status_is_fully_painted_at_narrow_width`,
`test_immersive_markdown_flavor_is_distinct_and_accessibly_painted[textual-dark]`
and `[textual-light]`, `test_selected_card_uses_the_bundles_focus_background`.
They had been measuring an unstyled layout.

AC#3 -- one re-pin, `test_grips_emit_correct_toggle_for_enter_space_and_pointer_click`:
`Button._on_click` DROPS a click while the button still carries `-active` from
its previous press (Textual's 0.2 s press animation). Traced with a wrapped
`_on_click`: at the click, `-active` was True on the branch and False on dev --
same geometry, same widget under the pointer, opposite outcome. The test now
waits that window out and keeps its three-toggle assertion.

Name-set comparisons (branch vs detached `origin/dev` ff2dc03145, same
selections): console batch 1 3=3, batch 2 5=5, batch 3 11 branch / 16 dev (the
five above), batch 4 11 branch / 0 dev -- of which 10 are
`test_schedules_workbench.py`, since shown to be broadly unstable on dev too
(4 -> 9 failures across two clean dev runs, with only one name in common), filed
as task-32502, and 1 is `test_library_core_loop_modes_are_actionable_without_
leaving_library`, which fails identically on dev when run alone.

Fix round 1 (task-8 review finding 3). The guard had two holes, both closed:
(a) the pin check was a substring test for `APP_STYLESHEETS`, so
`str(APP_STYLESHEETS[0])` -- the bundle alone, spelled differently -- passed;
the scan now folds a subscript of a real sheet sequence to the one path it
names before inlining (mutation re-run: that spelling on `_ChipsOverflowApp`
fails the guard, `fix2-guard-mutation-a.txt`); (b) the owner exemption matched
the whole FILE, so a module that merely imported `LibraryScreen` exempted
every harness in it -- it is now scoped to the harness class body plus its
same-module bases, the three screen-pushing harness bases are named as owners
(`ConsoleHarness`: all seven module-local ones push `ChatScreen` in
`on_mount`; `LibraryHarness`; `DestinationHarness`), and an imported
`OtherHarness.CSS_PATH` is read off the real class. The tightened scan run
against the pre-flip tree reported 18 more bundle-only harnesses in 14 files
(`fix2-guard-mutation-b.txt`); two more that the earlier rounds had listed,
`_ConversationCanvasHarness` and `_StyledManagerHost` (both spelled
`CSS_PATH = LibraryHarness.CSS_PATH`), were missing from that run because the
owner search matched the pin's own right-hand side -- all 20 (16 files) now
load `APP_STYLESHEETS`. Two
assertions that pinned the old spelling were re-pinned to the same set
(`test_skill_editor_production_geometry_contains_basic_and_advanced_workflows`:
`app.CSS_PATH == [str(p) for p in APP_STYLESHEETS]`;
`test_production_bundle_applies_speech_disclosure_styles`, whose selector
contract also moved from the bundle text to `app_css_text()` -- those rules
live in the settings split sheet now, and the node was red on the baseline
for exactly that reason). Name-set comparison of the 79 test functions that
reach the 20 harnesses (two chunks, branch vs detached HEAD 74db68de77):
chunk A 1 failed / 52 passed vs 11 / 42; chunk B 3 / 80 vs 14 / 69 before
the tts contract repair, and that node passes after it -- 21 baseline reds
fixed by the sheets, 0 introduced. The survivors fail identically on the
baseline: `test_raw_cli_collapsed_state_retains_danger_label_and_one_row_
geometry` (task-27018) and `test_high_stakes_file_notes_states_are_legible_
in_shipped_themes[size0|size1]` ('Save failed' at 3.89:1 on both trees).
Captures `fix2-chunk{A,B}-{branch,base}.txt`.

Files: `Tests/UI/test_consolidated_css_harness.py` (+ the scan and guard), 30
harness files (one-line pins), `Tests/UI/test_library_adaptive_reader_shell.py`
(the re-pin); fix round 1: 16 more test files (20 pins) and
`Tests/UI/test_settings_speech_tts_panel.py`'s text contract.

Landing (task-8 re-review, new MINOR): the owner search read the whole
harness class source, so `CSS_PATH = LibraryHarness.CSS_PATH` -- bundle-only,
no screen pushed -- exempted itself by naming the owner. It now collects the
names the class (and its same-module bases) references in code, minus the
`CSS_PATH` pin; docstrings and comments never count. Mutation on the branch:
`_StyledManagerHost` re-pinned that way passes the old guard
(`land-guard-mutation-c-old.txt`, `1 passed`) and fails the new one naming
the harness and `screen_agentic_library.tcss` (`land-guard-mutation-c.txt`,
`1 failed`); reverted, `6 passed` (`land-guard-green.txt`).

At the dev merge, dev had repaired `Tests/UI/test_settings_speech_tts_panel.py`
itself (4a02cab885: `_StyledPanelHarness` pins `[_BUNDLE, _SETTINGS_SHEET]` and
`test_production_bundle_applies_speech_disclosure_styles` reads the settings
sheet's own text); dev's spelling was kept over this branch's
`APP_STYLESHEETS` / `app_css_text()` version -- the harness loads the owning
sheet either way and the guard accepts both. Riders 32452/32453/32454 were
renumbered to 32501/32502/32503 at landing: dev already carried those ids
(MCP Hub UX waves A/B/C, merged and Done).
<!-- SECTION:NOTES:END -->
