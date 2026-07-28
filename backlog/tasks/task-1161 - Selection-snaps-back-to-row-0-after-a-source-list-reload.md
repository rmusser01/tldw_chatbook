---
id: TASK-1161
title: >-
  Selection snaps back to the first row after the source list reloads
status: To Do
assignee: []
created_date: '2026-07-28 12:00'
labels:
  - watchlists
  - bug
  - ui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After pressing **Check now**, the sources list reloads and the selection jumps back to the first row, losing whatever the user had selected. Seen in the live capture taken while verifying TASK-1105.

The cause is TASK-1100's fix: populating the table highlights row 0, and that highlight now selects. That behaviour is what arms `Check now` in the first place and should not simply be removed — but a *reload* is not a fresh population, and it should restore the selection the user already made rather than discard it.

Most visible immediately after a check, which is exactly when a user is likely to want to act on the source they just checked.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A source selected before a reload is still selected after it
- [ ] #2 A first population with nothing previously selected still highlights and selects row 0
- [ ] #3 A selected source that no longer exists after the reload degrades cleanly
- [ ] #4 A test selects a non-first row, triggers a reload, and asserts the selection survives, proven to fail against current code
<!-- AC:END -->
