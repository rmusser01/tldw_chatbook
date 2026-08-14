---
id: TASK-16224
title: Restore generated-video Loguru diagnostic rendering
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 08:40'
updated_date: '2026-08-14 08:44'
labels:
  - diagnostics
  - tests
  - privacy
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the generated-video retention warning so Loguru renders the sanitized exception type without exposing exception details.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The generated-video startup retention warning renders the exception type and no private exception text.
- [x] #2 The persistent diagnostic inventory matches the repaired source.
- [x] #3 Focused ProductionApp and architecture/static gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the literal %s output through the existing bounded-retention ProductionApp test
2. Restore Loguru-native placeholder formatting without adding exception capture
3. Regenerate and verify the governed persistent diagnostic inventory
4. Run focused privacy, architecture, and static gates; close the task

ADR required: no
ADR path: backlog/decisions/029-persistent-diagnostic-boundary.md
Reason: ADR-029 already governs the persistent diagnostic privacy boundary; this is a formatting bug within it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored Loguru-native `{}` formatting for the generated-video startup retention warning. The diagnostic still records only the exception type, with no exception capture or raw message. Regenerated the governed diagnostic manifest; exactly the existing app.py owner digest changed and owner/call/sink topology remained fixed.

Verification: the existing ProductionApp privacy regression changed from literal `%s` RED to 1/1 GREEN; non-write inventory verification passes; the full persistent-diagnostic architecture file passes 64/64; Ruff lint, py_compile, and diff checks pass. Ruff format remains red identically on HEAD and the repaired app.py due unrelated baseline formatting, so no broad app.py formatting churn was introduced.

ADR required: no. ADR-029 remains authoritative.
<!-- SECTION:NOTES:END -->
