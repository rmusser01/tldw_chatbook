---
id: TASK-40328
title: "A Library Collections rail press recomposes the whole screen three times, about 0.64-0.80 s of UI-loop CPU"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-13 07:06'
created_date: 2026-11-12 19:36
labels:
- follow-up
- library
- collections
- performance
priority: medium
references:
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/library_collections_controller.py
dependencies:
- TASK-39170
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (cred stream review r13c-rev2, measured on the Collections action node). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** One rail press costs 0.64-0.80 s of UI-thread CPU across three full Library rebuilds, each re-mounting the rail, canvas and reader: visible jank and focus churn on a common navigation step.

**Path** (`_select_library_rail_row_after_source_admission`): the route swap (`library_screen.py:23481`, `_replace_library_browse_canvas`) or a whole-screen `recompose()` (`:23484`), then `run_worker(self._load_library_collections_capture_entry())` (`:23504-23508`). That worker calls `_refresh_library_collections_capture_reader()` at `library_collections_controller.py:467`, again after `await controller.load_page` (`:503`), and again after `await controller.load_selected_now()` (`:506`). Each is `self.refresh(recompose=True)` (`:403-411`), and on this controller `refresh` is `Screen.refresh` (`:191-193`), so each is a whole-screen recompose. TASK-281 (the Library whole-screen-recompose umbrella) does not name this path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Collections rail press settles with at most one Library recompose; the capture loads update the reader pane in place
- [x] #2 The press's UI-loop CPU is measured before and after on the same host and recorded
- [x] #3 A test counts the recomposes of one press and is red before the fix
<!-- AC:END -->

## Implementation Plan

1. Re-check at base 26b2e677dd: count `compose_content` passes for one rail press and probe its UI-loop CPU (loop `thread_time`, several presses).
2. Add a pane-scoped repaint next to the panes (`Widgets/Library/library_collections_capture_reader.py`): hand the scope rows, Items and Work panes the new presentation and recompose only those whose presentation changed, re-focusing a removed control by id.
3. Route the entry loader's repaints through it, keeping the whole-screen recompose as the fallback when the panes are not mounted.
4. Test that one press makes at most one Library recompose and the panes show the loaded captures; re-measure; raise the controller's size pin by the measured growth only.

## Implementation Notes

- **Premise held at base 26b2e677dd** (2026-11-13): the entry loader still ended each step in `_refresh_library_collections_capture_reader`, a whole-screen `refresh(recompose=True)`. Counted on a seeded press: 4 `compose_content` passes (the route change plus three loader steps).
- **Fix.** `recompose_capture_panes` (in `Widgets/Library/library_collections_capture_reader.py`, beside the panes) hands the scope rows, Items and Work panes the new presentation and awaits a recompose of only the panes whose presentation changed; a control focused inside a rebuilt pane is focused again by id, with the mouse-capture release/sweep a teardown needs. The controller's `_sync_library_collections_capture_panes` captures the annotation draft, builds the presentation, falls back to the whole-screen recompose when the panes are not mounted, and re-resolves the reader layout after refresh; the loader's repaints go through it (four now: its two early `controller is None` exits folded into one). Other Collections actions still use the whole-screen helper (out of scope, see the report).
- **Tests.** `test_rail_press_recomposes_library_once_and_loads_captures_in_place` (`Tests/UI/test_library_collection_reader_journeys.py`, census node) counts `compose_content` from the press until the entry worker has finished and the Work pane shows the loaded capture: at most 1, both capture rows painted. Red with the fix reverted: "4 Library recomposes for one press". Because the route's shell now stays mounted while its panes repaint, tests that acted on the first shell they saw could press Quick Capture while it was still disabled (capabilities loading) or query a pane mid-recompose: `test_library_collections_capture_reader.py` gained `_wait_for_collections_entry` (waits for the entry worker) used by its four Quick Capture presses and by `test_library_collections_characterization.py`'s `_open_collections`.
- **Measured** (probe: Collections pressed from the landing, then 3 more times from Notes, UI-loop `time.thread_time()` from press to settled, `gc.collect()` first; base and head back to back, 2 launches each, load 37-45): base first press 1166-1193 ms, later presses 597-1181 ms (median about 780); head first press 779-836 ms, later presses 475-574 ms (median about 516).
- **Covering runs** (`-n 4`, temporary module-level `bootstrap_profile` marker for the plain files, removed afterwards): the seven Collections UI files plus `Tests/Architecture/test_library_collections_wiring.py` -- head 8 failed / 88 passed, every failure also red at base (compact-size journeys at 80x24); base had those plus `test_rail_count_never_falls_back_to_the_unfiltered_total_after_a_visit` (flaky) and the new test.
- **Lane.** New node in `scripts/ui_pr_gate_census.txt` (tail): 8.2 s call, 19.1 s total serial with the UI lane's flags at load 46.
- **Size.** `library_collections_controller.py` 1689 -> 1698 (+9, measured); pin raised by exactly that with a dated comment in `Tests/Architecture/test_library_modules_size_ratchet.py`. The pane logic went to the widget module to keep this small.
- Files: `tldw_chatbook/UI/Library_Modules/library_collections_controller.py`, `tldw_chatbook/Widgets/Library/library_collections_capture_reader.py`, `Tests/UI/test_library_collection_reader_journeys.py`, `Tests/UI/test_library_collections_capture_reader.py`, `Tests/UI/test_library_collections_characterization.py`, `Tests/Architecture/test_library_modules_size_ratchet.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
- **Fix round 1 (review r14ls-rev1, 2026-11-13).**
  - *Rail count and Details went stale* (task-32103's contradiction back): the loader's whole-screen recomposes had been the only thing repainting the rail row label and the Details sentence after the page loaded. After each in-place pane repaint, `_sync_library_collections_capture_panes` now re-syncs the rail in place from a fresh shell through `LibraryScreen._sync_library_rail_state`, a helper extracted from (and now used by) the entry and snapshot reconciles' identical `rail.sync_state(...)` calls. The Collections scope rows the rail's row-context factory builds now read the live presentation, so a rail-only recompose cannot rebuild them from the presentation captured at compose time. The rail-press test is parametrized: `late_capture` (a capture saved after the prefetch: the painted row must read "(3)") and `failed_prefetch` (the painted row must leave "(—)" for "(2)" and `#library-details-body` must drop "Collections count unavailable"); both read the painted widgets. Red on ec10ee34fa: "Rail row reads '▸ Collections (2)', the canvas (3)" and "Rail row reads '▸ Collections (—)', the canvas (2)"; green after (load 39-41).
  - *A cancelled repaint could strand a blank pane*: `presentation` was assigned before `await pane.recompose()`, so a cancel mid-recompose left the pane empty and an identical repaint skipped it. Each pane is now marked `_half_built` until its recompose completes, and a marked pane is rebuilt whatever its presentation. New census node `test_a_pane_repaint_cancelled_mid_recompose_is_rebuilt_by_the_next` (cancels the repaint task from inside the Items pane's recompose, then repaints with the same presentation): red without the mark ("An identical repaint left the pane with 0 rows"), green with it; 6.4 s call / 11.9 s total serial at load 34.
  - *Size.* `library_collections_controller.py` 1698 -> 1702 and `library_screen.py` 38164 -> 38165 lines, 1376 -> 1377 methods (measured; the screen helper replaced two 7-line calls); both pins raised by exactly that with dated comments. The rail-press node now runs 7.0 s + 6.6 s call / 21.3 s total serial at load 36.

