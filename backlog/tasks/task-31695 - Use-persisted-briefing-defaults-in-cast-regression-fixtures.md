---
id: TASK-31695
title: Use persisted briefing defaults in cast regression fixtures
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:37'
updated_date: '2026-09-05 18:46'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make cast fallback tests control the current call-time persisted-default resolver instead of an unused configuration alias.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fallback cases deterministically select their fixture provider through the canonical resolver.
- [x] #2 Explicit preset behavior and all existing script generation assertions remain covered.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the two fallback RED cases and inspect the current call-time persisted-default resolver. 2. Replace obsolete default_api_endpoint patches with the canonical resolver owner seam, retaining explicit-preset and generated-script assertions. 3. Run the complete briefing cast test file, scoped checks, parent review, and scoped commit. ADR required: no. ADR path: N/A. Reason: routine fixture correction to an existing resolver contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced three obsolete default_api_endpoint patches with the canonical briefing_service.resolve_persisted_briefing_defaults owner seam. The two fallback regressions failed before the fixture correction; explicit preset precedence and all existing provider/model/script assertions are unchanged. All 67 briefing cast tests passed within the clean 269-test combined gate (/private/tmp/tldw-31693-31695-final.xml). Scoped Ruff and diff checks passed. No production behavior changes or new ADR required.

Parent completed bounded final diff review with no actionable findings.
<!-- SECTION:NOTES:END -->
