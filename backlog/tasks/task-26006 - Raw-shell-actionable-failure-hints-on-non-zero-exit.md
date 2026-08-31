---
id: TASK-26006
title: 'Raw shell: actionable failure hints on non-zero exit'
status: To Do
assignee: []
created_date: '2026-08-31 15:44'
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
- [ ] #1 A non-zero exit whose output matches a known failure shape carries one appended, clearly-labeled hint
- [ ] #2 Hints are appended only on non-zero exit and never alter or truncate the original output
- [ ] #3 First match wins; at most one hint is added per invocation
- [ ] #4 Hint text is bounded and marked as tool-generated so it is never mistaken for command output
- [ ] #5 An unrecognized failure returns exactly today's output with no hint
- [ ] #6 The hint table is data, not branching logic, so adding a shape needs no control-flow change
<!-- AC:END -->
