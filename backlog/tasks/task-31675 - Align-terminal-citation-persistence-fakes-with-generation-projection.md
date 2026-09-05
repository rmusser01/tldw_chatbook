---
id: TASK-31675
title: Align terminal citation persistence fakes with generation projection
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:12'
updated_date: '2026-09-05 18:20'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore truthful terminal citation lifecycle tests under the generation projection persistence contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stopped failed and empty terminal cases retain their citation and persistence assertions
- [x] #2 Focused terminal citation tests pass with a current-contract persistence fixture
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test-only alignment with an existing persistence contract.
1. Reproduce failing terminal lifecycle cases and trace generation persistence requirements.
2. Adapt the smallest shared test fixture without weakening assertions.
3. Run the complete affected test file and static checks, then record evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The create-only call recorder now exposes the canonical projection reader contract, returning None because it owns no persisted canonical rows. Existing-row use therefore remains fail-closed; stopped/failed/empty paths retain every create, finalizer and transient-state assertion. An initial process-local marker hypothesis was rejected because it suppresses pending creates. No production or security behavior changed and no new ADR was needed. RED: five projection-unavailable failures. GREEN: terminal file 97 passed; combined citation files 223 passed in 126.62s. Ruff lint/format checks passed and diff self-reviewed.
<!-- SECTION:NOTES:END -->
