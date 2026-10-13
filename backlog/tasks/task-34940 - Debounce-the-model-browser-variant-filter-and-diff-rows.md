---
id: TASK-34940
title: Debounce-the-model-browser-variant-filter-and-diff-rows
status: To Do
assignee: []
created_date: '2026-10-13 14:58'
labels:
  - performance
  - ui-perf
  - wave-4-blocking-rebuilds
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The model-browser variant filter rebuilds the variant list on every keystroke: remove_children plus remount of every variant row widget and guidance recompute (model_remote_view.py:1527 -> :1138-1146), with no debounce. Bounded by variants-per-model (usually under 30) but it is the exact clear-then-repopulate-per-keystroke shape sibling views avoid.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Variant filter debounced and rows filtered in place or batched rather than full destroy/rebuild per keystroke
<!-- AC:END -->
