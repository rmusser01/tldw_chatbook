---
id: TASK-41504
title: "Library focus lands on a pane grip or the rail row instead of the first row in several code-driven and rail-click switches"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-18 18:15'
created_date: 2026-11-17 05:27
labels:
- follow-up
- library
- focus
- ui
priority: medium
references:
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/library_notes_controller.py
- Tests/UI/test_library_shell.py
dependencies:
- TASK-41128
- TASK-41129
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the libfocus stream and its review (TASK-41128 to TASK-41130 area, round 15); all measured as present on base too. Observed landings:
- in 1 of 4 wide probe runs a rail click left focus on the Notes rail row instead of the first note row;
- code-driven switches landing on a pane grip: landing then empty Notes `press()` at 170x48 lands on `#library-browse-library-grip`; Media then Collections `press()` lands on `#library-collections-library-grip`; Notes then resident Media `press()` lands on a grip about 1 in 3;
- a rail click Media then already-open Notes lands on the rail row (3 of 3 on base, 1 of 3 on head);
- at 170x48 rail entry into an empty Notes list sometimes leaves focus on the Library grip (libtests fix round);
- the Import once receipt's "view the imported notes" action still lands in the filter (the earlier line cite was wrong; find the receipt button by its copy in `library_notes_controller.py`), inconsistent with the TASK-41128 landing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each listed switch lands on its first row (or the documented empty-state control) 10 of 10 runs, and never on a pane grip
- [x] #2 The Import once receipt's view-imported-notes action lands where TASK-41128 decided Escape lands, or the difference is documented as a decision
- [x] #3 Tests drive each switch and assert the focused widget id
<!-- AC:END -->

## Implementation Plan

1. Probe every listed switch at the base (170x48, production CSS, loads 15-50) and record where focus lands.
2. Collections joins the shared list entry channel (task-2856) for a switch that leaves focus off the rail: its rows are `.library-collections-item-row`, the entry waits out the capture loader, and an empty page lands in the Filter captures box (the shared empty-list rule). A rail click or Enter keeps its rail row (TASK-40328, TASK-41130).
3. Import once's "View N imported notes" lands through `_land_on_library_notes_page` (TASK-41128's landing), not the filter.
4. Pin every listed switch in a PR-laned file, asserting the focused id, `screen.focus_chain` membership and no grip; show each red at the base or under a temporary source edit; run each 10 times.
5. Update the Collections and Notes guide pages; covering tests, ratchets, preflight.

## Implementation Notes

**Probe at the base** (`67583dfa72`, production CSS at 170x48, loads 15-50, 16 runs a case including a run with 20 CPU burners). The Notes and Media switches listed here already landed correctly 10 of 10 on this base: TASK-41129's rounds fixed them. Rail click into Notes, `press()` and click into an empty Notes list (from the landing and from Media), Notes -> resident Media `press()`, and Media -> resident Notes click all ended on the first row or the empty list's filter. Media -> Collections `press()` did not: it ended on `#library-ingest-top-button`, with or without saved captures, because Collections had no entry focus at all. The `#library-collections-library-grip` landing the task names showed up once Collections had one (see the loader wait below).

- **Collections joins the list entry channel** (task-2856) for a switch that leaves focus off the rail, such as the `press()` listed here. Its rows (`.library-collections-item-row`) are in `_LIBRARY_LIST_ROW_CLASS_BY_ROW_ID`, and a page with no capture lands in **Filter captures**, the rule every list but Media uses for an empty page.
- **A rail click or Enter on Collections still keeps focus on its rail row.** A first cut armed the entry for every Collections switch. The covering run then failed three pins in `test_library_collection_reader_journeys.py` (`test_collections_first_frame_opens_the_library_pane[rail_press]` and `test_rail_press_recomposes_library_once_and_loads_captures_in_place[failed_prefetch|late_capture]`), which hold the rail-row landing that TASK-40328 and TASK-41130 settled. The arm now runs only when focus was not on a rail row as the switch began, and those pins pass. No user-visible path changes, so the Collections guide page is unchanged. (TASK-41129 found the same for Media: a real rail click or Enter always moves focus to the rail first.)
- **The entry waits out the capture loader** (`group="library_collections_capture_entry"`). A trace showed why waiting for the controller's page state was not enough. `recompose_capture_panes` reads focus once, then awaits the scope rows, Items and Work panes one at a time. The 50 ms retry tick landed in the old Items pane while the scope rows recomposed, the Items recompose then removed the filter, and Textual's reset parked focus on the Collections Library grip (4 of 16 traced runs). Once the loader has finished, a later repaint finds focus set and restores it by id. Lesson added to `lessons-textual.md`.
- **AC#2.** "View N imported notes" now lands through `_land_on_library_notes_page`, as Import once's Escape and ‹ Notes do since TASK-41128: the first note row, or New with no note row.
- **Tests** (PR-laned files, no census edit):
  - `Tests/UI/test_library_rail_switch_keeps_reader_and_note.py::test_each_library_switch_lands_on_its_first_row_never_a_grip`, 8 cases (`rail-click-notes`, `press-empty-notes`, `click-empty-notes`, `media-click-empty-notes`, `media-press-collections`, `media-press-collections-with-captures`, `notes-press-resident-media`, `media-click-resident-notes`). Each case asserts the focused id, `screen.focus_chain` membership and no grip, after the entry window has closed and 0.5 s more.
  - `Tests/UI/test_library_notes_move_never_false_conflicts.py::test_view_imported_notes_lands_on_the_notes_page`.
- **Red evidence** (temporary edits, all reverted):
  - Base source: both Collections cases failed on `library-ingest-top-button` (2 of 2 runs each); the import view landed in `library-notes-filter`.
  - Base source without the rail-press arm: the five Notes cases failed at the switch (on the rail row, or `nav-home` for `press-empty-notes`).
  - `notes-press-resident-media` stays green without the arm, because the Media browse restores `#library-media-row-0` itself. Without both the arm and that `focus_identity` it failed 3 of 3 (focus None).
  - Head without the loader wait: both Collections cases failed 4 of 4 on `#library-collections-library-grip`.
- **10 of 10.** Final source, every new node of the three tasks (17) x 10 at `-n 6` with 16 CPU burners: 170 passed (load 14-43). Before the last two design changes, the 8 cases x 10 at `-n 4` gave 80 passed (load 16-28), and 160 of 160 passed with burners (load 25-52).
- **Re-pinned.** `Tests/UI/test_library_shell.py::test_background_recompose_restores_focus_on_a_route_without_a_list_row_class` used Collections as its route with no row class, the premise this task ends. It now uses Search, which still has none. Only its route, wait selector and messages changed, and it passes on head and base.
- **Covering runs** (final source, `-n 4`, a temporary local plugin adding the `bootstrap_profile` marker to plain Tests/UI nodes, then deleted):
  - **62 Library UI files** that reach the changed seams, plus `test_library_notes_w4_import_keyboard.py`: 51 failed, 1713 passed (load 12-23). All 51 also fail on the base source with the same command; `test_library_notes_folder_navigator.py::test_mounted_abandoned_receipt_reload_ignores_late_page_or_filter_result` failed 12 of 12 there too, in a 3-repeat run.
  - **`test_library_shell.py` + `test_screen_navigation.py`, `-k "note or focus or entry or collection or grip or rail or files"`** (401 nodes): 395 passed and 6 failed, then 4 after the re-pin above.
    - Fail on base too: `test_library_shell_note_save_then_back_refreshes_list_title_and_age` (deselected product defect) and the two `test_screen_navigation.py` `_focus_library_list_entry` stub nodes (`_FakeMediaRowButton` has no `is_attached`).
    - Flaky on both: `test_library_shell_blank_note_autosaved_then_emptied_still_gcs_on_back` (2 of 3 on head, 4 of 6 on base) and `test_library_note_newer_selection_cancels_pending_back_focus` (2 of 3 on head, 2 of 6 on base).
  - **Architecture ratchets and Library wiring** (serial): 321 passed. `Tests/Library` scope and row-walk and the diagnostic inventory: 7 failed, 194 passed; the same 7 fail on base (`RecoveryRequired`).
  - Ruff gate: no new findings. Preflight: passed.
- **Lane time.** The new nodes join two whole-file UI census entries, so the census is unchanged. Serial time with the UI lane's flags (load 38 falling to 15): `test_library_rail_switch_keeps_reader_and_note.py` +51.5 s across the 8 cases; `test_library_notes_move_never_false_conflicts.py` +61.5 s across 9 nodes (the three TASK-41600 cases are 12.5-13.0 s each). That is 112.8 s of wall time in all.
- **Size.** `library_screen.py` 38280 -> 38311 lines (1380 methods), `library_notes_controller.py` 7462 -> 7493, `canvas_sync.py` 1314 -> 1319: ratchet rows raised to the measurements, with notes.
- **Files.** `UI/Screens/library_screen.py`, `UI/Library_Modules/screen_constants.py`, `UI/Library_Modules/library_notes_controller.py`, `Tests/UI/test_library_rail_switch_keeps_reader_and_note.py`, `Tests/UI/test_library_notes_move_never_false_conflicts.py`, `Tests/UI/test_library_shell.py`, the three ratchet files, `Docs/User_Guide/library/notes.md`, `backlog/docs/lessons-textual.md`.
- **Docs.** `notes.md`: where the receipt's action leaves focus, and that opening an empty Notes list from the rail lands in the filter (the documented empty-state control; unchanged behaviour, now pinned).

**Fix round 1 (review r16fo-rev1, 2026-11-18).**
- **An empty Collections page lands on Quick Capture (coordinator's call, Minor 4).** With nothing narrowing the page, its own copy reads "press Quick Capture above", like Media's empty page, which lands on its action. After a scope or filter miss, or while Quick Capture is disabled, the page lands in the filter.
  - `library_collections_capture_reader.py` now names the two narrowings once (`_collections_scope_filters`, `_collections_scope_narrowed`, and the public `collections_scope_is_unnarrowed`), shared with `collections_empty_state_copy`.
  - `LibraryCollectionsController._library_collections_empty_entry_target` picks the control.
- **Tests.** `media-press-collections` now expects `library-collections-quick-capture`: red on `b0cd65a9b8`, which landed in `library-collections-filter`. The new `media-press-collections-empty-scope` sets the rail scope to Archived with two saved captures and expects the filter. It passed on `b0cd65a9b8`, and failed when the target was always Quick Capture. `test_library_crit10_pagers.py` still pins the copy rule.
- **Ceiling seen this round.** At load 59-69 (other agents), `media-press-collections-with-captures` ended with nothing focused, 2 of 2 runs: the capture loader outlasted the fixed 2 s entry window. That is TASK-32301's ceiling, a list arriving after the window; at load up to 52 it passed every run. Nothing changed for it here.
- **Proposed follow-up (Minor 2, not fixed).** At 80x24 a code-driven switch into Collections lands nothing, because its rows, filter and Quick Capture all sit in the collapsed Items pane. It is not a regression: the base ends on `nav-home`.
- **Size.** `library_screen.py` 38280 -> 38313 (1380 methods), `library_collections_controller.py` 1756 -> 1776. `canvas_sync.py` is back to 1314 and its row with it.
- **Runs (fix round 1, final source).**
  - **New and re-pinned nodes** (22, `-n 4`): 22 passed in each of 3 runs at load 12-24. An earlier run at load 72-83 failed 6 with nothing focused, all from entry windows expiring; the same command passed 21 of 21 at load 41-47.
  - **Covering set** (the 62 Library files plus `test_library_notes_w4_import_keyboard.py` and `test_library_crit10_pagers.py`, `-n 4`, with the temporary bootstrap-marker plugin): 60 failed, 1735 passed, at load 30-97 over 59 min.
    - 50 of the 60 are in the base-failing list.
    - Re-run on head at load ~30, the other 10 passed 9 of 10. The one failure is `test_mounted_abandoned_receipt_reload_ignores_late_page_or_filter_result[False-browse]`, which fails 12 of 12 on base.
  - **Serial ratchets and wiring** (now including the Collections wiring test): 325 passed. Preflight passed; ruff gate passed.

**Fix round 2 (review r16fo-rev2, 2026-11-18).**
- **Disabled Quick Capture falls back to the filter, now pinned.** The new case `media-press-collections-quick-capture-disabled` makes the scope service's `capabilities()` raise, so Quick Capture is drawn disabled, and expects `library-collections-filter`.
  - Red when the disabled check is dropped: focus None, because the entry kept aiming at a control outside the focus chain.
  - Red the same way when the target is always Quick Capture.
- **Margin for `media-press-collections-with-captures`.** Just before its switch, this case raises `LIBRARY_LIST_ENTRY_FOCUS_ARMED_SECONDS` to 6 s; no other case changes. Its serial call time goes from about 7.1 s to 10.8 s, about +3.7 s. The new disabled case adds 4.8 s.
- **Runs.** Both test files whole, `-n 4`, 3 runs: 30 passed each time (load 11-33). Serial ratchets and wiring: 325 passed. Preflight and the ruff gate passed.
