---
id: TASK-16852
title: 'Watchlists artifacts: script selection still rebuilds the scripts table inside the detail region'
status: Done
assignee: ['@claude']
created_date: '2026-08-16'
updated_date: '2026-08-16'
labels:
  - ui
  - watchlists
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-15779 (PR #1732) fixed briefing selection destroying the briefings table by moving
the detail chrome into a `BriefingDetailRegion` that recomposes alone. Its
Implementation Notes disclose the deliberately unexpanded scope — the same defect one
level down — and it still holds at dev `ee741cf10`:

Selecting a **script** rebuilds the WHOLE detail region, including the scripts table the
user is interacting with. `UI/Watchlists_Modules/artifacts_pane.py:1840-1842` —
`watch_selected_script` calls `_refresh_detail_region()` (`:1789`), which does
`region.refresh(recompose=True)`, and `#artifacts-scripts-table` is composed inside
`compose_briefing_detail()`, i.e. inside the region being torn down. So the 15779 bug's
symptom set (focused table destroyed under the user, scroll position lost, the
immediately-following arrow key press dead because the focused widget was unmounted)
recurs at the scripts level. The briefings table is unaffected either way — 15779's fix
stands.

Fix direction is the 15779 recipe applied one level down: split the script-detail chrome
into its own sub-region (or patch script-selection-dependent bits in place) so a script
selection updates the script detail without recomposing `#artifacts-scripts-table`.
The 15779 pin suite (`Tests/Watchlists/test_watchlists_artifacts_selection_in_place.py`)
is the template for the born-red evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Selecting a script updates the script detail without unmounting `#artifacts-scripts-table` (same widget instance, focus and scroll preserved; born-red test)
- [x] #2 An arrow-key press immediately after a script selection moves the selection on (the 15779 AC symptom, at the scripts level)
- [x] #3 The 15779 selection-in-place suite and the artifacts pane suites stay green
<!-- AC:END -->

## Implementation Plan

1. Re-locate the current line numbers for `watch_selected_script`/`_refresh_detail_region`/
   `compose_briefing_detail` in `artifacts_pane.py` (filing's line numbers may have drifted)
   and read `BriefingDetailRegion` + the 15779 Implementation Notes as the template.
2. Baseline `Tests/Watchlists/test_watchlists_artifacts_pane.py`,
   `test_watchlists_artifacts_selection_in_place.py`, `test_watchlists_scoped_rebuilds.py`
   at HEAD, to a file.
3. Add a nested stateless recompose boundary, `ScriptDetailRegion(RecomposeCaptureGuard,
   Vertical)`, mirroring `BriefingDetailRegion` one level down: it renders
   `ArtifactsPane.compose_script_detail()` (the script detail `Static` + the
   Synthesize/Play/Stop toolbar, extracted from `compose_briefing_detail`'s tail).
   `compose_briefing_detail` keeps building the Cast toolbar and the scripts `DataTable`
   directly (so they stay inside `BriefingDetailRegion`'s own recompose surface, as
   15779 left them), then yields `ScriptDetailRegion` instead of the old Static/toolbar
   pair.
4. `watch_selected_script` stops calling `_refresh_detail_region()`; instead it patches the
   scripts table's row highlight in place (a `_move_script_row_highlight`/
   `_restyle_script_row` pair mirroring `_move_briefing_row_highlight`/
   `_restyle_briefing_row`, sharing a new `_script_row_cells` helper with the table build)
   and calls a new `_refresh_script_detail_region()` that recomposes ONLY
   `ScriptDetailRegion`. `watch_script_audio` moves to the same narrower refresh (it only
   affects the script detail/Play button, never the scripts table). `watch_scripts`,
   `watch_scripts_with_audio`, `watch_citations` and `selected_briefing`'s own watcher keep
   calling `_refresh_detail_region()` (the whole region) unchanged, since they change what
   the scripts/citations table itself must show.
5. Born-red pins: a new `Tests/Watchlists/test_watchlists_artifacts_script_selection_in_
   place.py`, mirroring `test_watchlists_artifacts_selection_in_place.py`'s five shapes
   re-aimed at `#artifacts-scripts-table` (second-arrow-key movement, same-instance+focus,
   scroll preserved at depth, highlight moves in place, detail updates while the table's
   content stands still); plus a recompose-count pin in `test_watchlists_scoped_rebuilds.py`
   (script selection costs zero `BriefingDetailRegion` rebuilds, one `ScriptDetailRegion`
   rebuild).
6. Re-run the 15779/15778 suites (`test_watchlists_artifacts_pane.py`,
   `test_watchlists_artifacts_selection_in_place.py`, `test_watchlists_cold_read_swap.py`,
   `test_watchlists_scoped_rebuilds.py`) to confirm no regression; CSS: add a
   `#artifacts-script-detail-region` rule (the old `#artifacts-script-detail`'s `1fr` slot,
   now a container) and regenerate the bundle with `build_css.py`.
7. ruff check on touched files; update task notes; mark Done.

## Implementation Notes

Applied the 15779 recipe one level down, as a second nested recompose boundary rather
than a hand-rolled patch of every widget — commit to follow this note.

**Topology change.** `ScriptDetailRegion(RecomposeCaptureGuard, Vertical)` now nests
*inside* `BriefingDetailRegion`, holding just the script detail `Static`
(`#artifacts-script-detail`) and the Synthesize/Play/Stop toolbar — extracted verbatim
into a new `ArtifactsPane.compose_script_detail()` (the old tail of
`compose_briefing_detail()`). `compose_briefing_detail()` still builds the Cast toolbar
and `#artifacts-scripts-table` directly (unchanged from 15779: those stay inside
`BriefingDetailRegion`'s own surface), then yields `ScriptDetailRegion` instead of the
raw Static/toolbar pair. `selected_script`/`script_audio` no longer call
`_refresh_detail_region()`; `watch_selected_script` now takes `(old, new)` and calls a
new `_apply_script_selection_in_place()`, which patches the scripts table's row
highlight via `_move_script_row_highlight`/`_restyle_script_row` (mirroring
`_move_briefing_row_highlight`/`_restyle_briefing_row`, sharing a new
`_script_row_cells()` helper — the build and the patch read from the same source, same
discipline as `_briefing_row_cells`) and then calls a new
`_refresh_script_detail_region()`, which recomposes only `#artifacts-script-detail-region`
by id/type query. `watch_script_audio` moved to the same narrower call — it only affects
the script detail/Play button, never the scripts table (only `scripts_with_audio` does,
which stays on the wider `_refresh_detail_region()` path, unchanged). `watch_scripts`,
`watch_scripts_with_audio`, `watch_citations` and `selected_briefing`'s own watcher are
untouched: they still rebuild the whole `BriefingDetailRegion`, since a new `scripts`
list (or citations) changes what the SCRIPTS table itself must show.

**CSS.** New `#artifacts-script-detail-region` rule (`_watchlists.tcss`): `height: 1fr;
min-height: 1;` — the exact slot `#artifacts-script-detail` held directly as a sibling of
the scripts table before this task; that rule is unchanged, since it remains the sole
`fr`-declared child of its new immediate parent and still claims the same share. Bundle
regenerated with `build_css.py`; `check_bundle_sync.py` clean.

**Evidence.**
- Born-red pins (new `Tests/Watchlists/test_watchlists_artifacts_script_selection_in_
  place.py`, 5 tests mirroring the 15779 shapes, re-aimed at `#artifacts-scripts-table`):
  verified red at pre-fix HEAD (`094748b3e`, checked via a disposable
  `git worktree add --detach`) — all 5 fail for the destroy-rebuild reason: the AC
  symptom test asserts `3 == 2` (the second arrow key did nothing), the other four fail
  the `is`-identity check on the rebuilt table. All 5 green with the fix.
- A 6th pin added to `test_watchlists_scoped_rebuilds.py`
  (`test_a_script_selection_never_recomposes_the_briefing_detail_region`) counts
  `Widget.recompose` calls directly: a script selection now costs `ArtifactsPane`==0,
  `BriefingDetailRegion`==0, `ScriptDetailRegion`==1. Verified red at pre-fix HEAD too —
  there it fails at COLLECTION (`ImportError: cannot import name 'ScriptDetailRegion'`),
  since the class did not exist yet; a stronger signal than a graceful assertion
  failure, not a weaker one.
- Suites green post-fix: combined `-m ui` run of `test_watchlists_artifacts_pane.py` +
  `test_watchlists_artifacts_selection_in_place.py` (15779's 5 pins) +
  `test_watchlists_artifacts_script_selection_in_place.py` (this task's 5) +
  `test_watchlists_cold_read_swap.py` (15778's pins) = 141 passed, 15 deselected (no
  `ui` marker on that file's non-UI-only cases), 0 failed. `test_watchlists_scoped_
  rebuilds.py` 18/18 (15461/15779's 17 + this task's 1 new). `Tests/UI/test_watchlists_
  destination_shell.py` 80/80 (15779's own geometry/shell baseline, unchanged). Collect-
  only sweep of `Tests/Watchlists`: 700 tests, no import breakage (694 baseline + 6 new).
- ruff check clean on all touched files (`artifacts_pane.py`, both test files).

**Files.** `tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py` (nested region +
watchers + in-place patching; module/reactive docstrings updated),
`tldw_chatbook/css/features/_watchlists.tcss` + regenerated `tldw_cli_modular.tcss`, new
`Tests/Watchlists/test_watchlists_artifacts_script_selection_in_place.py`, updated
`Tests/Watchlists/test_watchlists_scoped_rebuilds.py`.
