---
id: TASK-31960
title: >-
  Library media - Review these is the only list-wide action outside the
  failed-load gate
status: To Do
assignee: []
created_date: '2026-09-07 08:27'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
J final review M1: every list-wide action routes through the failed-load gate except 'Review these', which keeps _gate_stale_action alone, so on a failed first page it stands live beside a dimmed Export. It is defensible - its worker re-fetches and notifies on failure - but the asymmetry is unexplained at the surface and is one line to remove if symmetry is what we want.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The list-wide actions present one consistent enabled state on a failed first page, or the exception is documented where the gate is defined
- [ ] #2 The decision is recorded either way
<!-- AC:END -->
