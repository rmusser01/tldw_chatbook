---
id: TASK-31706
title: Initialize privacy diagnostics in the partial Console migration fixture
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:50'
updated_date: '2026-09-05 18:57'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the minimal historical migration fixture prerequisites without changing hostile schema validation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All partial v41 migration shapes reach and preserve the actual migration validation;The complete migration file and privacy checks pass with no production change.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce hostile and permitted v41 partial-shape fixtures failing before validation on missing privacy diagnostic identity. 2. Initialize the existing diagnostic hash exactly as the normal DB constructor does while retaining the minimal in-memory schema fixture. 3. Run all migration shapes and owner privacy checks; retain exact reject/accept and rollback assertions. ADR required: no. ADR path: N/A. Reason: test-only constructor prerequisite, no schema or diagnostic boundary change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Initialized the minimal memory-only fixture diagnostic identity to memory, matching the real constructor; no runtime schema or logging changes. This exposes all historical hostile/default/constraint validation paths rather than failing before migration begins. Complete migration, SQLite owner privacy and roots gate files:159 passed/3.18s (/private/tmp/tldw-review-migration-roots-final-20260905.xml). Ruff, scoped formatting and diff checks pass; exact rollback and validation assertions retained. ADR not required: test-only constructor prerequisite.
<!-- SECTION:NOTES:END -->
