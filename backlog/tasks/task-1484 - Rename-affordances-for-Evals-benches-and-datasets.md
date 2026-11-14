---
id: TASK-1484
title: >-
  Rename affordances for Evals benches and datasets
status: Done
assignee: []
created_date: '2026-07-30 10:00'
labels:
  - evals
  - word-bench
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from live UAT (2026-07-30). Datasets are created as "Untitled dataset <hex>" and benches as "<name> <hex>"; nothing in the UI can rename either, so the catalog fills with hex-suffixed placeholders. Detail panes are read-only Statics by design in the shipped slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] A dataset can be renamed from its detail pane
- [x] A bench can be renamed from its detail pane
- [x] Rail rows and grid headers reflect the new name immediately
<!-- AC:END -->

## Implementation Notes

Implemented in the Evals UX follow-up batch (fix/t1-evals, 2026-10-11).

- **Benches**: the bench editors ARE the rename affordance -- both
  `BenchEditor` (`#evals-bench-name`) and `CharacterBenchEditor`
  (`#evals-cb-name`) carry a name Input, and their Save posts `Saved`,
  which the screen answers with a rail-dirty re-selection. This landed
  with task-1482/task-1691's editors after this task was filed; the
  2026-10-11 premise-verify confirmed both, so no new bench code was
  needed.
- **Datasets**: new `DatasetRenameRow` (`snippet_editor.py`) -- a
  prefilled Input + Rename button stacked in a Vertical (an unstyled
  Input in a one-line Horizontal overflows the pane and parks the
  button under the inspector column; confirmed live via pilot click
  resolution, which is why the stack needs no styling). Mounted under
  the heading in BOTH dataset detail surfaces: `SnippetEditor` and
  `ProbeSetDetail` (a probe set IS a dataset row; its probe listing
  stays read-only). A successful rename posts `DatasetRenamed`, handled
  screen-side with `select(kind="dataset", rail_dirty=True)` -- rail row
  and heading re-render from what `EvalsDB.update_dataset` persisted,
  immediately. Blank names warn; UNIQUE collisions (no `deleted_at`
  exemption) toast and keep the original name.
- **Grid headers**: a bench rename reaches a run's grid header through
  that run's snapshot by provenance design -- historical runs keep the
  name they ran under, new runs carry the renamed bench. That is the
  intended `word_bench.storage` contract ("results render from a
  snapshot taken at launch, never from the live task"), not a gap.

Modified files: `tldw_chatbook/UI/Evals/snippet_editor.py`,
`tldw_chatbook/UI/Evals/character_bench_editor.py`,
`tldw_chatbook/UI/Screens/evals_screen.py`,
`Tests/UI/test_evals_snippet_editor.py` (3 tests),
`Tests/UI/test_evals_empty_states.py` (probe-set rename test).
