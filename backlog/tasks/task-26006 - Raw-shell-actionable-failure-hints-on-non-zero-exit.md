---
id: TASK-26006
title: 'Raw shell: actionable failure hints on non-zero exit'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:44'
updated_date: '2026-09-01 18:04'
labels:
  - tools
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A failed shell command returns raw stderr and nothing else. Verified on origin/dev: a named grep for hint, suggest and recovery across Tools/raw_cli_executor.py returns zero - output is passed through unchanged. Hermes maps known non-zero-exit output shapes to a single actionable recovery line (first match wins, non-zero exits only), which saves the agent a diagnostic round-trip on the common failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A non-zero exit whose output matches a known failure shape carries one appended, clearly-labeled hint
- [x] #2 Hints are appended only on non-zero exit and never alter or truncate the original output
- [x] #3 First match wins; at most one hint is added per invocation
- [x] #4 Hint text is bounded and marked as tool-generated so it is never mistaken for command output
- [x] #5 An unrecognized failure returns exactly today's output with no hint
- [x] #6 The hint table is data, not branching logic, so adding a shape needs no control-flow change
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: first-match/bounded/labeled, nonzero-only, data-table shape, provider append-order + no-hint cases\n2. FAILURE_HINT_TABLE (8 rows, data) + failure_hint() in raw_cli_executor\n3. Provider _tool_result appends the hint after the untouched detail
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
FAILURE_HINT_TABLE in raw_cli_executor.py: 8 (pattern, hint) rows — command-not-found, permission-denied, missing-path, missing-Python-module, network refusal, not-a-git-repo, shell-syntax, port-in-use; failure_hint(exit_code, output) returns '[tool hint] …' for the FIRST match on non-zero integer exits only, None otherwise (AC#2/#3/#5; the label is AC#4's tool-generated marker and every hint is a fixed <200-char string). Applied in the provider's _tool_result: appended AFTER the untouched detail block (pinned: original stderr intact, hint strictly after it, exactly one). Note the detail cap slices BEFORE the hint is appended, so a hint can push the model-facing result slightly past _MAX_MODEL_RESULT_CHARS — bounded by the fixed hint length, chose not to re-slice so the hint can never be truncated into ambiguity. 7 new tests; raw-shell suites at the exact 3-name baseline.
<!-- SECTION:NOTES:END -->
