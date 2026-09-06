---
id: TASK-31559
title: Refresh Console transcript geometry ownership assertions
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 01:38'
updated_date: '2026-09-05 01:40'
labels:
  - console
  - tests
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the two Console internals tests that still assert pre-overflow-hint ownership after the tab strip moved into a dedicated row.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Transcript title is asserted against the center panel content region.
- [x] #2 Tab-row styling is asserted on the current owning row rather than its scroll child.
- [x] #3 Both exact regressions and focused nearby layout contracts pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Compare both failing geometry assertions with the current `ConsoleSessionSurface` composition and TASK-28028 overflow-hint ownership.
2. Assert the title against the padded content region and the visual tab role against the dedicated row while retaining child-strip compatibility checks.
3. Run both exact regressions, nearby transcript layout tests, Ruff, and diff checks.

ADR required: no
ADR path: N/A
Reason: this updates stale assertions to the existing Console session-surface composition; it changes no UX or runtime boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Updated the top-edge assertion to use the padded center panel's content region and moved height, margin, background, and transcript-offset assertions to the dedicated tab row introduced by TASK-28028. The scroll child remains addressed by its stable ID.
- Evidence: both CI regressions and two neighboring session-surface/transcript contracts pass 4/4; Ruff and diff checks pass.
- ADR required: no; test-only alignment with the shipped composition.
<!-- SECTION:NOTES:END -->
