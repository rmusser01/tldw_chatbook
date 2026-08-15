---
id: TASK-16275
title: Make Library ingest integration tests capability-independent
status: Done
assignee: []
created_date: '2026-08-14 21:14'
updated_date: '2026-08-14 21:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove developer-environment dependency from Library ingest integration coverage while preserving its forecast, consent, and persistence contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ingest option persistence expects the complete current generic option snapshot.
- [x] #2 Forecast tests deterministically exercise missing optional tooling regardless of installed extras.
- [x] #3 Local consent coverage deterministically exercises the missing-OCR path.
- [x] #4 Focused Library ingest integration coverage passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce and classify the Library ingest failures against current option defaults and installed optional dependencies.
2. Update fixtures to force the intended capability states and current persisted option shape.
3. Run focused and adjacent Library integration verification.

ADR required: no
ADR path: N/A
Reason: deterministic test-fixture maintenance for existing behavior; no runtime boundary changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Expanded the exact persisted generic ingest-option expectation for the current snapshot fields and replaced assumptions about which optional packages happen to be installed with one scoped capability fixture. The forecast and consent tests now force only their required feature gaps, so they exercise the same product paths in minimal and fully provisioned developer environments. The four original Library failures reproduced before the change; the full Library ingest module and Console companion passed 14 tests, and the exact 25-file regression slice passed with 292 tests and 6 optional-dependency skips. Ruff check and diff-check passed; Ruff format remains inherited-red on both touched test modules, confirmed against HEAD, so no unrelated whole-file formatting churn was introduced. ADR required: no; runtime behavior is unchanged.
<!-- SECTION:NOTES:END -->
