---
id: TASK-16210
title: Recognize Loguru format arguments in log hygiene test
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 00:14'
updated_date: '2026-08-14 00:14'
labels:
  - test-health
  - logging
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the missing-f-string logging guard focused on truly unbound brace placeholders instead of rejecting Loguru's supported positional and keyword formatting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plain brace templates with no formatting values remain rejected.
- [x] #2 Loguru templates with positional or keyword formatting values are accepted.
- [x] #3 The complete hygiene file, mutation, static, containing-chunk, and diff gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this corrects a static-test oracle without changing logging policy or production behavior.

1. Preserve the five valid QwenCloud Loguru calls rejected by the current oracle as RED evidence.
2. Characterize broken, positional, and keyword template shapes.
3. Report only brace templates with no positional or keyword formatting values, then run focused/chunk/static/diff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Narrowed the AST oracle to unbound plain brace templates. Constant Loguru templates with positional or keyword formatting values are now accepted, while the original missing-f-string shape remains rejected. RED: the sweep incorrectly reported five QwenCloud calls that each supplied their required values. GREEN: three hygiene tests and the 25-file containing chunk (204 tests) passed. Removing the positional-argument guard failed the new discriminating fixture. Removed the file's pre-existing unused `pytest` import so scoped Ruff is green; format and diff checks passed. ADR required: no; production logging is unchanged.
<!-- SECTION:NOTES:END -->
