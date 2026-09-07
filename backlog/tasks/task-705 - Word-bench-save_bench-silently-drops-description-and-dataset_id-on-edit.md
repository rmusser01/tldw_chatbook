---
id: TASK-705
title: >-
  Word bench save_bench silently drops description and dataset_id on edit
status: Done
assignee: []
created_date: '2026-07-26 14:30'
labels:
  - evals
  - word-bench
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the whole-branch review of PR 2 of the Evals rebuild (the word bench engine). Not a defect introduced by that PR unless stated; each is a seam the engine leaves for the screen that consumes it.

`storage.save_bench(..., task_id=...)` persists name, prompt mode, top-K, probes, and targets. `description` and `dataset_id` are dropped with no error.

The spec says the bench editor sets "name, description, dataset, prompt mode, top-K, probe list, and targets", so PR 3 will render controls for two fields that do not save. This is the same silent-discard footgun already recorded for `Evals_DB.update_run`, one layer up.

`update_task` already accepts `description`. Changing a bench's dataset needs a small `Evals_DB` addition, or an explicit decision that a bench's dataset is immutable after creation — either is fine, but it should be a decision rather than an omission.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] `save_bench`'s edit path persists `description`
- [x] Either `dataset_id` is persisted on edit, or the docstring states the dataset is immutable after creation so PR 3 can disable the control
- [x] A test round-trips an edited description
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in PR #924. `save_bench`'s edit path now persists `description`.

`dataset_id` was made explicitly **immutable after creation** rather than editable. Extending `update_task` to accept it looked trivial but broke existing edit-path tests with a real foreign-key failure, so the constraint is documented in `save_bench`'s docstring instead of half-implemented. PR 3 should disable the dataset control on an existing bench.
<!-- SECTION:NOTES:END -->
