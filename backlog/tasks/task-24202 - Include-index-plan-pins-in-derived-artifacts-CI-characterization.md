---
id: TASK-24202
title: Include index-plan pins in derived-artifacts CI characterization
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29 13:23'
updated_date: '2026-08-29 13:24'
labels:
  - ci
  - tests
  - database
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Align the required derived-artifacts workflow test with the shipped index-plan pin checker so the resilience assertion counts and validates all independent checker steps.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The derived-artifacts test inventory includes scripts/check_index_plan_pins.py
- [x] #2 All six checker steps remain asserted to run after an earlier failure
- [x] #3 The complete derived-artifacts workflow test module passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: test inventory maintenance for an already-shipped CI checker; no workflow or database policy change. Add scripts/check_index_plan_pins.py to CHECKERS, make checker-count wording data-driven/current, run the exact failure and full CI module, then Ruff/format/compile/diff checks and close.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the derived-artifacts workflow characterization to include the already-shipped scripts/check_index_plan_pins.py checker and replaced stale fixed-count wording with inventory-driven wording. The workflow itself was correct: all six checker steps already use !cancelled(), so no CI behavior changed. ADR required: no; ADR path: N/A. Verification: complete Tests/CI/test_derived_artifacts_workflow.py 7 passed in 0.67s; Ruff check passed; Ruff format check passed; compileall passed; git diff --check passed.
<!-- SECTION:NOTES:END -->
