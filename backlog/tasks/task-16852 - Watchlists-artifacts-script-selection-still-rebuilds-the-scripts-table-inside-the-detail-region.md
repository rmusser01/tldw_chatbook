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
the script detail/Play button, never the scripts table. `watch_scripts`/`watch_citations`
and `selected_briefing`'s own watcher are untouched: they still rebuild the whole
`BriefingDetailRegion`, since a new `scripts` list (or citations) is a genuine row-SET
change to the SCRIPTS (or citations) table.

**Review round.** Initial pass left `watch_scripts_with_audio` on the same wide
`_refresh_detail_region()` path, on the same "changes what the table must show"
reasoning applied to `scripts`/`citations` above — review caught that the reasoning does
not actually hold for it: `scripts_with_audio` never adds or removes a script row, it
only changes an EXISTING row's Audio-column cell (`_script_row_cells` is the only reader,
via `self.scripts_with_audio.get(row.get("id"))`). Concretely, a first Synthesize for the
SELECTED script — the pane's own primary action on a script, and the single most natural
next step after the very selection this task just fixed — lands its status through
exactly this reactive (`_synthesize_audio`'s `finally` unconditionally reloads via
`_load_briefings` → `_apply_briefing_state_to_pane` → `pane.scripts_with_audio = ...`),
so the WHOLE-region rebuild was reopening the destroy-rebuild defect on the feature's own
happy path. Fixed by patching every row's cells in place instead
(`watch_scripts_with_audio` now loops `self.scripts` and calls `_restyle_script_row` per
row, the same helper `watch_selected_script` already uses) — the scripts table's mounted
identity is never touched by an audio-status change. The Synthesize/Play/Stop toolbar
needs no extra wiring for this: it renders from `script_audio` (the selected script's own
newest render), never from `scripts_with_audio`, and `watch_script_audio` already
refreshes `ScriptDetailRegion` independently.

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
  **correction (review round): the failure mode is a normal runtime
  `AssertionError: assert 1 == 0` on the `BriefingDetailRegion` recompose count** (a
  script selection recomposed that region once, pre-fix), not the `ImportError` the first
  pass of these notes claimed. `ScriptDetailRegion` is never imported as a Python name in
  that test file — it only appears as a string dict key for the `Counter`-based
  `_RebuildCounter`, so a missing class could never raise `ImportError` there even in
  principle; `Counter.__getitem__` on an absent key returns `0` instead. Reproduced
  directly (a disposable `git worktree add --detach 094748b3e`) before writing this
  correction. Still genuinely born-red, and arguably a *more* legible failure than the
  originally-claimed one (it names the exact "must never rebuild the WHOLE detail region"
  violation) — only the originally-written mechanism was wrong.
- A 7th pin, `test_a_scripts_with_audio_change_patches_the_audio_cell_without_rebuilding_
  the_table` (added to `test_watchlists_artifacts_script_selection_in_place.py` in the
  review round): selects a script, focuses the table, then assigns `pane.scripts_with_
  audio = {selected_id: STATUS_COMPLETE}` — the exact reactive write `_apply_briefing_
  state_to_pane` performs once a Synthesize worker's reload lands. Verified red at the
  pre-review-fix commit (`dc0a05e42`, disposable worktree): `AssertionError` on the
  table's `is`-identity, same shape as the other six. Green post-fix, and the Audio
  cell for the synthesized row shows `ArtifactsPane._AUDIO_GLYPH` in place.
- Suites green post-fix: combined `-m ui` run of `test_watchlists_artifacts_pane.py` +
  `test_watchlists_artifacts_selection_in_place.py` (15779's 5 pins) +
  `test_watchlists_artifacts_script_selection_in_place.py` (this task's 6, after the
  review round's 7th pin) + `test_watchlists_cold_read_swap.py` (15778's pins) = **142
  passed**, 15 deselected (no `ui` marker on that file's non-UI-only cases), 0 failed.
  `test_watchlists_scoped_rebuilds.py` is `pytest.mark.asyncio`, NOT `ui` (a `-m ui`
  filter deselects it silently — run it unmarked): **18/18** (15461/15779's 17 + this
  task's 1 new). `Tests/UI/test_watchlists_destination_shell.py`: **80/80** (15779's own
  geometry/shell baseline, unchanged). `check_bundle_sync.py`: 5/5 clean (no CSS touched
  in the review round). Collect-only sweep of `Tests/Watchlists`: 700 tests at the initial
  pass, no import breakage (694 baseline + 6 new); +1 more test added in the review round
  (the 7th pin), no further breakage.
- ruff check clean on all touched files (`artifacts_pane.py`, both test files), both
  passes.

**Files.** `tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py` (nested region +
watchers + in-place patching; module/reactive docstrings updated),
`tldw_chatbook/css/features/_watchlists.tcss` + regenerated `tldw_cli_modular.tcss`, new
`Tests/Watchlists/test_watchlists_artifacts_script_selection_in_place.py`, updated
`Tests/Watchlists/test_watchlists_scoped_rebuilds.py`.
