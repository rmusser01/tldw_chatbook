---
id: TASK-34940
title: Debounce-the-model-browser-variant-filter-and-diff-rows
status: Done
assignee: []
created_date: 2026-10-13 14:58
labels:
- performance
- ui-perf
- wave-4-blocking-rebuilds
dependencies: []
priority: low
updated_date: 2026-10-13 19:11
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The model-browser variant filter rebuilds the variant list on every keystroke: remove_children plus remount of every variant row widget and guidance recompute (model_remote_view.py:1527 -> :1138-1146), with no debounce. Bounded by variants-per-model (usually under 30) but it is the exact clear-then-repopulate-per-keystroke shape sibling views avoid.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Variant filter debounced and rows filtered in place or batched rather than full destroy/rebuild per keystroke
<!-- AC:END -->


## Implementation Notes

- `Input.Changed` on the variant filter now debounces 0.2s (repo set_timer/partial idiom): the clear+remount of the variant rows runs once typing pauses, rapid changes cancel earlier timers, and `Input.Submitted` bypasses the debounce. Scope note: the row rebuild is still a destroy/remount when it runs (bounded by variants-per-model, usually <30) — what shipped is the debounce, not in-place row diffing.
- Tests: `Tests/UI/test_model_variant_filter_debounce.py` (debounced rebuild, rapid-change cancellation, unresolved no-schedule). 3/3 pass. The model-remote suite sits at its pre-existing error baseline.
- ADR required: no — input debounce in one view.