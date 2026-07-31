---
id: TASK-1480
title: 'Disambiguate Evals rail rows: runs get status glyph and timestamp'
status: Done
assignee: []
created_date: '2026-07-30 10:00'
updated_date: '2026-07-31 01:46'
labels:
  - evals
  - word-bench
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by live UAT (2026-07-30). After one sample bench, the rail shows three near-identical rows: bench `loaded-nouns (sample) fbf8b2d0`, dataset `loaded-nouns (sample) 3a7644f6`, run `loaded-nouns (sample) fbf8b2d0` — the bench and run rows are byte-identical. The blocked-reason copy relies on the user distinguishing bench rows from run rows, and nothing visually distinguishes them; hex suffixes are the only difference and they carry no meaning.

The design spec's rail mock renders run rows as `● 14:31 run` / `✓ 14:02 run` / `✗ 13:55 run` — status glyph plus start time.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Run rows render a textual status marker plus start time (spec mock shape), not a bare copy of the bench name; status is never conveyed by color alone
- [x] #2 Bench, dataset, and run rows are distinguishable at a glance without reading hex suffixes
- [x] #3 Tests assert the run-row label format for completed, failed, and cancelled runs
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Roll up per-group status inside run_groups()'s existing pivot pass
2. Render run rows as <glyph> <HH:MM> · <name> with single-width glyphs
3. Defensive timestamp parse; escape the interpolated name
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Commit 059b3725c. run_groups() rows carry "status" (any running -> running; else any cancelled -> cancelled; else completed), computed in the same single pass — no extra DB reads on the rail's compose path. Run rows render `● 14:31 · name` / `✓ …` / `✗ …` per the spec mock; glyphs are verified single-cell-width (cell_len == 1, never emoji), names are escape_markup-ed (Button labels parse markup), created_at parses defensively with a raw-string fallback. Deliberate ruling, surfaced for product review: an all-cells-failed run renders ✓ (DB status "completed") — the grid's failure callout owns that explanation; a distinct glyph would need per-group cell scans in a per-recompose path. Verified live: bench and run rows are now visually distinct; the second (successful) run appeared as ✓ 20:47 above the failed ✓ 20:30.

TASK-1480 amendment (user-directed, commit 059b3725c's follow-up on the same branch): reversed this task's own "completed always renders the done glyph" ruling above. New scheme -- running -> ● (unchanged); completed with >=1 successful cell -> ✓ (unchanged, partial failures still ✓); completed with ALL cells failed (>=1 cell) -> ✓✗ (new); cancelled, or run-level status "failed" (the eval_runs CHECK constraint allows it even though WordBenchRunner never writes it) -> ✗ (folded into the same bucket "cancelled" already used). Precedence: running > cancelled/failed > completed. Zero-cell completed group stays ✓ (vacuously nothing failed). Needed cell-level data list_runs() doesn't carry, so added EvalsDB.run_group_cell_failure_counts() -- one aggregate GROUP BY over eval_results joined to eval_runs on run_group_id, read once per run_groups() call (never a per-group query loop), returning {group_id: (total_cells, errored_cells)}; the failure/success discriminator mirrors word_bench.storage._cell_from_payload's own "error" key check via SQLite json_extract rather than a second definition. run_groups() gained "all_cells_failed"; library_rail.py's glyph selection split into _run_group_row_glyph. Commit on this branch (see git log for the exact SHA -- filed after this task's own commit).
<!-- SECTION:NOTES:END -->
