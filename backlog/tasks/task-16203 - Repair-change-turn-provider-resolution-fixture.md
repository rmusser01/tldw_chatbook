---
id: TASK-16203
title: Repair change-turn provider-resolution fixture
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 06:24'
updated_date: '2026-08-14 06:24'
labels:
  - test-health
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore change-turn tracking coverage on the current Console provider contract so the real bridge run reaches the tracking behavior instead of failing inside an obsolete resolution double.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Change-turn bridge tests use a contract-valid provider resolution and continue to exercise real tracking behavior.
- [x] #2 The focused change-turn suite passes with a mutation proving the obsolete resolution shape fails.
- [x] #3 Scoped lint and diff checks pass, with no new formatting drift or production changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this is a test-fixture correction for an existing Console provider boundary.

1. Reproduce the stale provider-resolution failure and compare the fixture with current bridge tests.
2. Replace the opaque object with the smallest contract-valid ConsoleProviderResolution.
3. Run the focused file, restore the obsolete double to prove RED, then run scoped static and diff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the obsolete opaque provider-resolution object with the current minimal `ConsoleProviderResolution` contract, allowing every change-turn bridge scenario to reach its real tracking assertions, and removed two stale unused imports exposed by the scoped lint gate. RED mutation: restoring `object()` made the representative disk-change test fail because the run ended before any snapshot was recorded. GREEN: all 35 change-turn tests passed; scoped Ruff check and diff-check passed. Ruff format would reformat the file identically on HEAD, so the small repair preserves inherited formatting drift rather than churning the whole test module. ADR required: no; test-fixture repair only.
<!-- SECTION:NOTES:END -->
