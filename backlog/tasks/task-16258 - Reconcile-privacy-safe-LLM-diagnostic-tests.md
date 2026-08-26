---
id: TASK-16258
title: Reconcile privacy-safe LLM diagnostic tests
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 16:14'
updated_date: '2026-08-14 16:21'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Align LLM curated-install diagnostic tests with the governed privacy-safe persistent logging contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Curated preflight and installation failures assert constant operation labels plus safe exception type without artifact identifiers.
- [x] #2 Malformed curated references remain contained without private worker details or invented identity values.
- [x] #3 Focused LLM diagnostics tests and static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a test-only reconciliation to the existing persistent-diagnostic privacy boundary.

1. Preserve the three deterministic RED assertions from the checkpoint.
2. Replace private artifact-identity expectations with exact safe label/type and negative privacy assertions.
3. Run focused and complete nearby LLM tests, Ruff, formatter characterization, and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced stale artifact-identity log assertions with exact constant operation-label and exception-type assertions plus negative checks for artifact fields and private worker details.
- Kept malformed-reference containment coverage while asserting no invented `unknown` identity is persisted.
- The three focused regressions and all 91 LLM adoption tests pass; Ruff lint/format and diff hygiene pass. The five Ollama probe tests also pass when their documented loopback access is permitted.
<!-- SECTION:NOTES:END -->
