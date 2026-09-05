---
id: TASK-31709
title: Restore explicit Change Review enablement in workspace roots fixtures
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
Exercise root validation after intentionally disabled binding setup using current persisted review preferences.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Enabled consumers prove exact root prefilters and disabled gates short circuit;Complete roots file and scoped static checks pass without changing runtime authority.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace empty roots to persisted workspace review being disabled during binding setup; global environment enablement alone does not opt that workspace in. 2. Explicitly enable only the fixture workspace through its existing registry API and assert disabled roots before enabling. 3. Preserve exact root prefilters and both enable-gate short circuits; full-file tests and scoped static checks. ADR required: no. ADR path: N/A. Reason: test-only existing authority setup; no production permission or ownership change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Binding setup under disabled Change Review persisted disabled workspace consent. Explicitly opted the fixture workspace in through the existing registry method after asserting that disabled state; global and per-workspace short circuits and exact ro/rw prefilter assertions remain unchanged. Complete roots, migration and owner privacy files:159 passed/3.18s. Ruff, scoped formatting and diff checks pass; no production authority change. ADR not required: fixture-only explicit consent.
<!-- SECTION:NOTES:END -->
