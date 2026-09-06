---
id: TASK-31762
title: Repair screen navigation regression cluster
status: Done
assignee: []
created_date: '2026-09-05 05:23'
updated_date: '2026-09-05 08:36'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and repair current screen navigation lifecycle, focus, and route regressions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduced screen navigation failures pass
- [x] #2 Screen navigation module passes in full
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the screen-navigation failures and group them by lifecycle/focus/route contract. 2. Repair stale tests or production behavior with the smallest justified change. 3. Run focused regressions and the full screen-navigation module. ADR required: no. ADR path: N/A. Reason: this is localized regression maintenance for existing navigation behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired the 28-failure screen-navigation cluster by updating stale Files, Collections, focus, route, and stack-ownership test contracts to the current shell behavior. Made the shared app factory keep splash disabled through deferred compose so developer-local settings cannot delay or destabilize route pilots. Fixed a real rapid-navigation race by making Schedules workbench refresh workers tolerate unmounts before touching descendant widgets.

Verification: Tests/UI/test_screen_navigation.py — 142 passed; focused Schedules error paths plus rapid-switch regression — 3 passed; Ruff on all changed Python files and git diff --check passed.

ADR required: no. ADR path: N/A. Reason: localized regression maintenance preserving existing navigation and scheduling boundaries.
<!-- SECTION:NOTES:END -->
