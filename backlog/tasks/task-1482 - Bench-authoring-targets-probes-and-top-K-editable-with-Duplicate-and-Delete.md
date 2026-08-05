---
id: TASK-1482
title: 'Bench authoring: targets, probes, and top-K editable with Duplicate and Delete'
status: Done
assignee: []
created_date: '2026-07-30 10:00'
updated_date: '2026-07-31 14:15'
labels:
  - evals
  - word-bench
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from live UAT (2026-07-30). Three of the five results-grid lenses can never show real data through the shipped UI: the Probe lens reports "no probes configured" and there is no probe authoring; the Δ baseline lens and spread sort need two or more targets and there is no target picker. The only reachable bench is the hardwired single-target sample, so the analysis engine's cross-target features are complete but unreachable. Imported datasets are equally stranded: the blocked copy says "select a bench that uses this dataset instead", but no bench can ever be pointed at one.

The design spec already covers this (bench editor mock with `[ + Add target ]`, probes row, and `[ Duplicate ] [ Delete ]` inspector actions); it was deferred out of the vertical slice. This is the largest remaining gap between the shipped screen and the approved design. Needs its own plan; not part of the 1476-1481 fix batch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A bench's targets can be added and removed from the bench editor (target picker over configured models)
- [x] #2 Probes and top-K are editable on a bench
- [x] #3 A bench can be created against any existing dataset (closing the imported-dataset dead end)
- [x] #4 Duplicate and Delete exist for benches per the spec's inspector actions
- [x] #5 The Probe and Δ baseline lenses are reachable with real data through UI-authored benches
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Bench editor field form: name/description/prompt mode/top-K/probes editable, Save/Revert, BenchEditor.Saved -> screen re-selects (Task 5).
2. Selection-yank guard so a completing background worker never recomposes over unsaved editor state the user has navigated into (Task 2, sequenced early since Task 5 depends on it).
3. duplicate_bench in storage.py, concurrency-copy semantics, _unique_name relocated into the engine layer (Task 3).
4. "+ New bench" in the rail: draft bench bound to the selected-or-newest dataset, zero-target Run gate (Task 4).
5. Bench editor targets become editable: staged target list, per-row Remove, Add picker over llama_cpp eval_models, zero-models "Create target" affordance posting CreateTargetRequested for the screen to handle (Task 6).
6. Duplicate/Delete buttons in the inspector pane, single-flight delete confirm, spec compose order (Task 7).
7. End-to-end test through both cross-target lenses: import -> "+ New bench" -> author (create-target path + Add picker, probes, top-K) -> Save -> Run -> Probe and Δ baseline lenses render real per-target-distinct data; backlog closeout (Task 8).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: eight sequenced tasks turned the previously-hardwired single-target sample bench into a fully authorable one, then proved the whole loop end to end.

Seams shipped:
- Field form (bench_editor.py): name/description/prompt-mode/top-K/probes editable, display-only until Save; failure renders in-place (#evals-bench-form-error), never a recompose that would discard unsaved typing.
- Target editing: a staged _staged_target_ids list mutated by per-row Remove and an Add picker (Select over EvalsViewModel.llama_targets()) via targeted remove_children()+mount_all() on #evals-bench-targets-section only. When zero llama_cpp eval_models rows exist anywhere, the picker is replaced by #evals-bench-create-target, which posts BenchEditor.CreateTargetRequested for evals_screen.py to handle (bench_editor.py must never import the capture client/runner, even transitively -- a source-scan test pins this) -- the screen resolves/creates the row via sample_bench.resolve_sample_target(create=True, name=BENCH_EDITOR_TARGET_NAME) and calls the widget's public stage_target().
- "+ New bench" in the rail: a draft eval_tasks row (target_ids=()) bound to the selected-or-newest dataset, no provider gate (a plain DB write, unlike the sample bench). A zero-target Run gate (UI reason + engine RuntimeError belt) stops the resulting dead-end-toast pattern.
- Duplicate/Delete in the inspector: storage.duplicate_bench (concurrency-copy semantics -- copies travel with the bench); a single-flight delete-confirm flag (a plain bool, not exclusive=True -- push_screen_wait's asyncio.shield makes cancelling the worker actively wrong, see evals_screen.py's own docstring) guards a real reentrancy the reviewer reproduced (two queued presses pushing two ConfirmationDialogs).
- Selection-yank guard (_selection_unmoved_since_launch): a completing background worker (sample-bench or bench-run) only auto-navigates to its fresh run group when the screen's selection is unchanged since launch, or has since moved into that same bench's own run groups -- otherwise a toast only, never a recompose that would blow away an editor mid-edit.
- Cross-worker guards: three flags/exclusive-groups close the gap where the sample-bench worker and the bench-run worker (each in its own Textual exclusive group) could both be in flight at once.

Key decisions:
- Dataset-fixed-at-create (deviation from a "fully editable bench" reading of the ACs): save_bench has no dataset_id parameter -- eval_tasks.dataset_id is a real FOREIGN KEY, and re-pointing an already-run bench at a different dataset would leave its own run history referencing snippets that may no longer exist under the new dataset's ids. AC #3 ("created against any existing dataset") is satisfied at CREATE time (the rail's dataset-selection binding); re-pointing an existing bench was ruled out of scope for this program, not silently dropped.
- Duplicate's screen-side catch is broad Exception, not duplicate_bench's own narrower RuntimeError -- a corrupt legacy bench's load_bench call can raise a plain ValueError downstream that a narrow catch would miss (Task 3 review ruling, later confirmed live).
- Add picker is deliberately NOT staged-filtered -- an already-staged target is still a selectable option, rejected inline ("Target already on this bench.") only on click, matching the create-target button's own "reuse rather than special-case" convention.

Environment migration: a TCC denial on ~/Documents (mid-program, persistent) moved the program to a standalone clone (/private/tmp/tldw-recovery, this branch) off an independent venv (/private/tmp/tldw-venv); granular Task 1-5 history survives in the original checkout's .git (inaccessible until TCC is restored) but this branch's own commits are the working truth going forward -- see progress.md's own ENVIRONMENT MIGRATION entry.

Task 8 (this task): Tests/UI/test_evals_authoring_e2e.py -- one test drives the full loop against a real in-memory EvalsDB and the real EvalsScreen: rail Import (a 4-snippet plain-text file, via the established _handle_dataset_import_file_selected test convention) -> "+ New bench" -> the create-target button (zero llama_cpp rows) -> an intermediate Save (needed to persist the first target and recompose, which is what makes a second, distinct target reachable from the Add picker at all -- the create-target button can only ever mint a SECOND target by reusing the first one, since sample_bench.resolve_sample_target reuses any existing llama_cpp row) -> the Add picker for a second, directly-seeded target -> rename/probes/top-K -> final Save -> Run (via a fake capture client, screen._sample_bench_client_factory) -> the rail's "✓" glyph -> the grid's Probe lens (real, per-target-distinct readings, one target hits both configured probes, the other reads "never observed") and Δ baseline lens (a real, nonzero Spread column). The fake client is the one deliberate departure from every other Evals worker test's shared-distribution fake: it returns genuinely DIFFERENT top-K distributions keyed by target name, which is what makes the cross-target lenses non-vacuous. Confirmed the test fails on a broken seam before confirming it passes (temporarily short-circuited the CreateTargetRequested handler's stage_target call; reverted via git checkout after). Full sweep green: Tests/UI/test_evals_*.py + Tests/Evals (830 passed, 13 skipped, 1 pre-existing unrelated failure -- test_evaluation_metrics.py's float-precision assert, already flagged in progress.md's Task 6 entry, not touched by this program) plus the fspicker/CSS-bundle-sync guards.

Modified/added files (this task): Tests/UI/test_evals_authoring_e2e.py (new); this task file.
<!-- SECTION:NOTES:END -->
