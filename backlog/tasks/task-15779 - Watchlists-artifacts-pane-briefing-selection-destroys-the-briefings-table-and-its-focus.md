---
id: TASK-15779
title: 'Watchlists artifacts pane: briefing selection destroys the briefings table and its keyboard focus'
status: Done
assignee: ['@claude']
updated_date: '2026-08-16 09:30'
created_date: '2026-08-13 12:31'
labels:
  - bug
  - watchlists
  - ux
priority: medium
---

## Description

Pre-existing UX defect found and recorded ("worth its own task") in
task-15461's Implementation Notes (input-latency burn-down's Watchlists
scoped-rebuild work). Task-15461 reduced the artifacts pane's recompose count
on a briefing selection from 2 down to 1, but that remaining recompose still
rebuilds the pane wholesale: selecting a briefing recomposes `ArtifactsPane`,
which destroys and rebuilds the briefings `DataTable`, which loses keyboard
focus. A second arrow-key press then does nothing until the user manually
re-focuses the table — measured on the pre-task-15461 code too, so this is
not something task-15461 introduced, just something it exposed by getting
the recompose count down to a level where the remaining one is now the
visibly broken step.

## Acceptance Criteria

- [x] Selecting a briefing in the artifacts pane does not destroy the
      briefings `DataTable` widget instance (in-place update instead of a
      recompose that tears it down), OR the recompose explicitly restores
      focus and cursor position to the table afterward
- [x] A second arrow-key press immediately after selecting a briefing moves
      the cursor to the next row (regression test — this is the concrete
      symptom: today it does nothing until the user re-focuses)
- [x] The scripts/audio/citations clearing task-15461 folded into
      `watch_selected_briefing` (via `set_reactive`) keeps working unchanged
- [x] `Tests/Watchlists/test_watchlists_artifacts_pane.py` stays green

## Implementation Plan

1. Baseline `test_watchlists_artifacts_pane.py`, `test_watchlists_cold_read_swap.py`
   (task-15778's 16 pins) and `test_watchlists_scoped_rebuilds.py` at HEAD, to a file.
2. Born-red pins (new `Tests/Watchlists/test_watchlists_artifacts_selection_in_place.py`):
   selecting a briefing keeps the SAME `#artifacts-table` widget instance, its focus,
   its scroll position, and a second arrow key immediately moves the cursor (the AC's
   concrete symptom); the selected row's highlight moves in place; the detail region
   shows the newly selected briefing's body; the table's painted content is otherwise
   identical before/after.
3. Decompose the recompose surface: everything below the briefings table ("Briefing
   detail" title, `#artifacts-detail`, citations table, scripts/audio section) moves
   into a new `BriefingDetailRegion(RecomposeCaptureGuard, Vertical)` child that owns
   NO state — it renders from the parent pane's reactives via
   `ArtifactsPane.compose_briefing_detail()`, and is the ONLY thing rebuilt when a
   selection-derived value moves.
4. Flip the six selection-derived reactives (`selected_briefing`, `scripts`,
   `selected_script`, `script_audio`, `scripts_with_audio`, `citations`) to
   recompose=False; their watchers refresh the region (coalesced by Textual's
   `_recompose_required` flag) and, for `selected_briefing`, also update in place the
   two selection-dependent toolbar buttons (Export/Keep) and the table's
   selected-row style/cursor — the table widget itself is never torn down.
5. CSS: `#artifacts-detail-region` takes over the detail half's share of the pane's
   `fr` pool (8fr against the table's 2fr, mirroring the old flat 2:6:1:1 split);
   regenerate the bundle with `build_css.py`; re-run the pinned geometry tests.
6. Update `test_a_briefing_selection_costs_one_pane_recompose` (task-15461's pin of
   the then-best-possible 1) to pin the new topology: 0 pane recomposes, the region
   rebuilt at most once, clearing semantics unchanged.
7. ruff check + format on touched files; full three-file suite re-run; task notes.

## Implementation Notes

Took the AC's first branch (in-place update; never tear the table down),
implemented as a decomposition of the recompose surface rather than a
hand-rolled patch of every widget — commit `f6154560d`.

**Topology change.** Everything below the briefings table ("Briefing
detail" title, `#artifacts-detail`, citations table, scripts/audio
section) moved into `BriefingDetailRegion(RecomposeCaptureGuard,
Vertical)` — a stateless recompose boundary: it owns no reactives and
renders from the parent pane's state via the new
`ArtifactsPane.compose_briefing_detail()` (the old `compose` tail,
verbatim). The six selection-derived reactives (`selected_briefing`,
`scripts`, `selected_script`, `script_audio`, `scripts_with_audio`,
`citations`) flipped to `recompose=False`; their watchers refresh only
that region (`refresh(recompose=True)`, coalesced by Textual's own
`_recompose_required` flag, so a selection plus its reload landing costs
at most one region rebuild per pump drain). `watch_selected_briefing`
additionally patches, in place, the only selection-dependent chrome
outside the region: the table's selected-row highlight
(`update_cell` by row/column key, cells built by the same
`_briefing_row_cells` `compose` uses, so the two paths cannot drift; the
cursor follows a programmatic selection by row only, never yanking the
user's column) and the Export/Keep buttons (shared
`_export_button_state`/`_keep_button_state`). Pane-level
`recompose=True` remains for the reactives that change what the table
itself shows (`briefings`, scope/toolbar/picker state) — those rebuilds
still reseed the cursor as before. The task-15461 synchronous clearing
(`_clear_selection_derived_state`, `set_reactive`) is unchanged; it now
rides the region refresh instead of a pane recompose.

**Why not "recompose + restore focus" (the AC's OR branch):** restoring
focus/cursor onto a rebuilt table still loses scroll and column state and
would need the same treatment again for the reload that lands
scripts/citations a moment later; the boundary fixes both arrivals at
once and follows the seed-vs-watch discipline 15775/15778 established.

**CSS.** New `#artifacts-detail-region` rule: `8fr` against the table's
`2fr` (the old flat 2:(6+1+1) split, now two-deep — the region's children
re-divide 6:1:1 inside it), `min-height: 8`, `overflow-y: auto`
(belt-and-suspenders one level down from the pane's own rule). Bundle
regenerated with `build_css.py`; `check_bundle_sync.py` clean. All four
pinned geometry tests re-run green.

**Evidence.**
- Born-red pins (new `Tests/Watchlists/test_watchlists_artifacts_
  selection_in_place.py`, 5 tests, real key presses through the real
  focused table): all red at HEAD `637cb3892` for the destroy-rebuild
  reason — the AC symptom red with the selection stuck (second press did
  nothing), the other four red on widget identity — and all green with
  the fix. They pin: second-arrow-key movement; same-instance + focus;
  scroll preserved at depth (40 rows, parked at row 34); highlight moves
  AND leaves in place; detail/citations correctness after the reload
  lands, with the table's painted content byte-identical and the Export
  button armed in place.
- `test_a_briefing_selection_costs_one_pane_recompose` (task-15461's pin
  of the then-best-possible 1) renamed/updated to
  `test_a_briefing_selection_never_recomposes_the_pane`: pane == 0,
  region == 1, clearing assertions unchanged.
- Suites green post-fix: `test_watchlists_artifacts_pane.py` 131/131
  (AC #4; includes the AC #3 clearing pin
  `test_switching_the_selected_briefing_clears_stale_scripts_before_the_
  reload_lands`); `test_watchlists_cold_read_swap.py` (task-15778's 16
  pins) + `test_watchlists_scoped_rebuilds.py` 32/32;
  `Tests/UI/test_watchlists_destination_shell.py` 80/80. Collect-only
  sweeps: Tests/Watchlists 694, Tests/UI 12,819 — no import breakage.
  Baselines at HEAD before the change: 131 + 32 passed.
- ruff check clean on all touched files; the new test file ruff-formatted
  (the three pre-existing files were not format-clean at HEAD, so
  whole-file reformatting was deliberately skipped as diff noise).

**Files.** `tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py`
(region + watchers + in-place patching; module docstring updated),
`tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (two stale
comments only — no code), `tldw_chatbook/css/features/_watchlists.tcss` +
regenerated `tldw_cli_modular.tcss`, new
`Tests/Watchlists/test_watchlists_artifacts_selection_in_place.py`,
updated `Tests/Watchlists/test_watchlists_scoped_rebuilds.py`.

Known, deliberately unexpanded scope: selecting a SCRIPT still rebuilds
the scripts table (it lives inside the region), exactly as it did
pre-task — the same defect one level down, but outside this task's AC;
the briefings table now survives script selections too, which is a
strict improvement.
