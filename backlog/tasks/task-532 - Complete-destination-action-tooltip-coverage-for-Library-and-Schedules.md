---
id: TASK-532
title: Complete destination action tooltip coverage for Library and Schedules
status: Done
assignee: []
created_date: '2026-07-24 19:55'
updated_date: '2026-07-24 20:04'
labels:
  - ui
  - accessibility
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure every actionable control in the active Library and Schedules destinations explains its outcome so users can understand controls before activation and the destination-wide accessibility audit remains green.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Library rail-top ingest action exposes a concise, non-empty tooltip.
- [x] #2 The Schedules owner-selection and error-clearing actions expose concise, non-empty tooltips.
- [x] #3 Schedules enables and describes server ownership only when an active server ID and scheduling service are both available.
- [x] #4 The destination-wide audit reports all missing tooltip button IDs for a route in one failure.
- [x] #5 Focused Library, Schedules, and destination-wide tooltip tests pass.
- [x] #6 Formatting, linting, and diff-integrity checks pass for the changed files.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused assertions that document the expected tooltip contract for the uncovered actions.
2. Add outcome-oriented tooltip copy to the Library and Schedules controls.
3. Make the destination audit aggregate missing controls so one run exposes the complete gap.
4. Run focused tests, the destination-wide audit, and static checks.
5. Request an independent code review and resolve any actionable findings.

ADR required: no
ADR path: N/A
Reason: This is small UI accessibility polish within existing screen contracts and does not change architecture, storage, ownership, security, or cross-module interfaces.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added outcome-oriented tooltips to the Library ingest action and the Schedules owner, sync-error, and conflict-resolution actions.
- Made Schedules server ownership availability require both an active server ID and the notifications-service wrapper, consistently across render, refresh, and click handling.
- Changed the destination tooltip audit to report every missing button ID for a route in one failure.
- Added exact-copy and state-transition regression coverage; the affected Library, Schedules, and destination-shell modules pass with 154 tests and one pre-existing Personas skip.
- Ruff lint, Ruff formatting, and `git diff --check` pass for all changed Python files.
- Independent review found the server-availability inconsistency; the fix was re-reviewed and approved with no remaining actionable findings.
- ADR required: no. This remains UI accessibility polish within existing application boundaries.
<!-- SECTION:NOTES:END -->
