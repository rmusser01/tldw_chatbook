---
id: TASK-533
title: Measure visible navigation geometry at the overflow boundary
status: Done
assignee: []
created_date: '2026-07-24 20:07'
updated_date: '2026-07-24 20:12'
labels:
  - ui
  - tests
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the shell navigation overlap regression aligned with the scroll-strip clipping model so it detects visible collisions without failing on clipped child geometry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The default-width navigation regression compares the overflow hint with the visible scroll-strip boundary.
- [x] #2 The regression still verifies that the visible portion of Settings cannot cross into the overflow hint.
- [x] #3 The focused navigation test and master-shell navigation tests pass.
- [x] #4 The full visual-parity module is replayed and any unrelated failures are captured in scoped follow-up tasks.
- [x] #5 Ruff lint, formatting, and diff-integrity checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Replace raw Settings child-region comparison with scroll-strip and clipped-visible-region assertions.
2. Run the focused failure, master-shell navigation tests, and full destination visual-parity module.
3. Capture unrelated visual-parity failures in scoped follow-up tasks.
4. Run static checks and request independent review.

ADR required: no
ADR path: N/A
Reason: This only corrects a stale test’s geometry model and does not change runtime behavior or application boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Updated the default-width regression to compare the destination scroll strip with the overflow hint and to intersect Settings with the strip’s visible content region.
- Confirmed the rendered geometry is strip columns 0–125, hint columns 126–139, and clipped Settings columns 119–125; the raw Settings child region beyond the strip is not painted.
- The focused regression and master-shell navigation suite pass with 18 tests.
- Replayed the full visual-parity module: 71 tests pass and the 12 unrelated stale contracts are fully captured in TASK-534 (five Watchlists cases) and TASK-535 (seven Schedules cases).
- Ruff lint, Ruff formatting, and `git diff --check` pass.
- Independent review approved the geometry correction and the follow-up task coverage with no remaining actionable findings.
- ADR required: no. This is a test-contract correction only.
<!-- SECTION:NOTES:END -->
