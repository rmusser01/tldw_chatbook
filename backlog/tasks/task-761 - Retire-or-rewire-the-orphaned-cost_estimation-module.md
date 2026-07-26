---
id: TASK-761
title: >-
  Retire or rewire the orphaned Utils/cost_estimation module
status: To Do
assignee: []
created_date: '2026-07-26 17:10'
labels:
  - evals
  - cleanup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during PR 3a's Task 4 review, while adjudicating why the bench editor's cost estimate reads "not estimated yet" rather than a figure.

`tldw_chatbook/Utils/cost_estimation.py` exists, carrying a `MODEL_COSTS` table and `estimate_evaluation_cost()` whose docstring is "Cost estimation utilities for evaluation runs." It now has **zero importers**: its only consumer was `Widgets/Evals/cost_estimation_widget.py`, deleted in PR 1 (#922) as part of the unreachable Evals UI. The same orphan pattern as the ~8,800 lines that PR retired.

It was deliberately **not** wired into the new bench editor, and that was the right call: its `MODEL_COSTS` table has no entry for `gpt-4o-mini` or `gpt-3.5-turbo-instruct` — the models this feature's own fixtures use — so it falls through to a generic $0.01/1k default. Rendering that as a dollar figure would be a confidently wrong number, which is worse than an absent one in a tool whose purpose is measuring model behaviour accurately.

So there are two honest options and one dishonest one. Either retire the module as dead code, or refresh its price table and wire it in behind a check that a model is actually present rather than defaulted. What must not happen is wiring it in as-is.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] A reachability check confirms `Utils/cost_estimation.py` still has no importers at implementation time
- [ ] Either the module is deleted, or its `MODEL_COSTS` table is refreshed and it is wired into the Evals bench editor's estimate
- [ ] If wired in, a model absent from the table renders as unknown rather than silently using the generic default
- [ ] If deleted, the deletion guard in `Tests/UI/test_evals_deletion_guard.py` gains its path
<!-- AC:END -->
