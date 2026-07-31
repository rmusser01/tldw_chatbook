---
id: TASK-1610
title: Run completion must not destroy a dirty bench editor
status: Done
assignee: []
created_date: '2026-07-31 15:10'
updated_date: '2026-07-31 19:09'
labels:
  - evals
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Whole-branch review of the bench-authoring program (task-1482), Important 1. The selection-yank guard added for run completion checks selection identity only: when a sample or bench run completes while the user is still ON the launched bench, the completion select() recomposes the screen and destroys every typed field and staged target in the editor. The sample-worker case is the sharpest: "Create sample bench" is always available in the rail, and its completion yanks to a run group belonging to a DIFFERENT bench than the one being edited. Fix shape: BenchEditor grows a dirty flag; `_selection_unmoved_since_launch` (or the workers' completion paths) consults it and degrades to the existing "— see the Runs section." toast when the mounted editor is dirty.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A run completing while the mounted bench editor holds unsaved edits never recomposes the screen
- [x] #2 The clean-editor and moved-selection behaviors are unchanged
- [x] #3 Tests cover the dirty-editor case for both workers
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read BenchEditor's field/save logic and evals_screen.py's `_selection_unmoved_since_launch` plus both worker completion paths to confirm the exact seam.
2. Extract the probes-line parse in `_on_save_pressed` into a shared module-level `_parse_probes_text` helper so Save and the new dirty check can never disagree.
3. Add `BenchEditor.is_dirty()`: computed on demand (no reactive watchers) from the same five widgets Save reads, plus the staged target list, compared against `self._loaded_config`.
4. Wire the guard: `_selection_unmoved_since_launch` queries the mounted `#evals-bench-editor` defensively (QueryError -> no editor) and returns False whenever it is dirty, overriding both of its existing "safe" branches -- this covers both `_run_bench_worker` and `_create_sample_bench_worker` from one seam.
5. Write red-first-equivalent tests: two worker-level dirty-editor tests (bench-run, sample-bench), one clean-parked-on-an-unrelated-bench regression test, and a unit suite for `is_dirty()` (pristine, each field, whitespace-only probe line, trailing empty line, staged add/remove, save-reload, never-composed).
6. Run the full evals UI test files foreground, commit, then run the mutation check (is_dirty() forced False) to confirm the new worker tests actually catch the regression, then restore.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: `_selection_unmoved_since_launch` gated `select()` purely on selection identity -- unchanged-selection or drilled-into-own-run-group both counted as "safe", even when the mounted BenchEditor held unsaved form state. The sample-bench worker's completion is the sharpest case: its launch selection can be ANY bench the user happened to be parked on (the button is a persistent rail affordance, unrelated to the bench it creates/runs), so an unmoved-but-dirty editor for a totally different bench was still getting yanked.

Fix: `BenchEditor` grows `is_dirty()` -- computed on demand (no reactive watchers; nothing here posts a live `Changed` message) by re-reading the same five widgets `_on_save_pressed` reads plus the staged target list, and comparing each against `self._loaded_config`. Probes go through a newly extracted `_parse_probes_text` module function shared verbatim by Save and `is_dirty()`, so the zero-length-line-drop / whitespace-only-line-keep rule can never diverge between the two. `_selection_unmoved_since_launch` now queries `#evals-bench-editor` defensively (QueryError -> no editor mounted) and returns False whenever it is dirty, overriding both of its existing "safe" branches -- one seam, so both `_run_bench_worker` and `_create_sample_bench_worker` are covered without duplicating the query.

Clean-editor and moved-selection behavior is unchanged: is_dirty() reads False the instant a bench is freshly selected, immediately after Save reloads, and for a widget that never composed a form at all (no db / unreadable row -- `_loaded_config` stays None).

Tests: two worker-level dirty-editor regressions (bench-run, sample-bench) in test_evals_screen.py, one clean-parked-on-an-unrelated-bench test proving the dirty check doesn't over-fire, and a 9-test is_dirty() unit suite in test_evals_bench_editor.py (pristine, each scalar field incl. an unparseable Top-K, whitespace-only probe line, trailing-empty-line no-op, staged add/remove, save-then-reload, never-composed). Mutation check: forcing is_dirty() to return False unconditionally failed exactly the two new worker-level tests (test_bench_run_completion_does_not_yank_a_dirty_bench_editor, test_sample_bench_completion_does_not_yank_a_dirty_bench_editor) and left everything else green -- confirmed then reverted via Edit. Full suite (Tests/UI/test_evals_bench_editor.py + test_evals_screen.py + test_evals_empty_states.py): 183 passed.

Modified files: tldw_chatbook/UI/Evals/bench_editor.py, tldw_chatbook/UI/Screens/evals_screen.py, Tests/UI/test_evals_bench_editor.py, Tests/UI/test_evals_screen.py.
<!-- SECTION:NOTES:END -->
