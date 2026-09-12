---
id: TASK-467
title: 'Internal Prompts panel: test that a subsystem header hides when search filters all its rows'
status: Done
assignee:
  - '@zcode'
created_date: '2026-07-22 22:10'
labels:
  - internal-prompts
  - test-coverage
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
InternalPromptsPanel._on_search hides a subsystem's group header when all its rows are filtered out (any_visible flag). The behavior is correct by inspection but has no direct test. Add one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test searches for a term matching only one subsystem and asserts the non-matching subsystems' group headers have display=False
- [x] #2 The matching subsystem's header remains visible
<!-- AC:END -->

## Implementation Plan

1. Add a test to Tests/UI/test_internal_prompts_panel.py reusing its _Host harness: search a needle matching exactly one subsystem's prompts, assert that subsystem's #group-header-<name> stays visible and every other subsystem's hides.

ADR required: no
ADR path: N/A
Reason: Test-only coverage addition; no behavior change.

## Implementation Notes

Added ``test_search_hides_group_headers_of_fully_filtered_subsystems`` to ``Tests/UI/test_internal_prompts_panel.py``. Needle "rewind" matches only ``console.rewind_summarize``, so the test asserts the console group header keeps display=True while all six other subsystem headers go display=False -- both AC clauses (non-matching hidden, matching visible) in one sweep over ``authoring.iter_specs_by_subsystem()``. Behavior was already correct by inspection, as the task stated; this pins it. All four internal-prompts test files pass together (12 passed). No production change.
