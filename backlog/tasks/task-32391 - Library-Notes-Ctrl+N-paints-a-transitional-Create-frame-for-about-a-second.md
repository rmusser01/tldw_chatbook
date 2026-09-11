---
id: TASK-32391
title: 'Library Notes: Ctrl+N paints a transitional Create frame for about a second'
status: To Do
assignee: []
created_date: '2026-09-11 10:30'
labels:
  - library
  - notes
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pressing Ctrl+N on a loaded profile briefly renders an intermediate "Create..." frame before the draft editor appears -- about a second under load. It resolves on its own, so nothing is lost, but the flash reads as a failed press and invites a second Ctrl+N. Observed during the task-32356 round on the seeded profile.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Ctrl+N goes from the list to the draft editor without painting an intermediate frame, or the intermediate frame is a deliberate labelled loading state
- [ ] #2 A second Ctrl+N during the transition does not create a second draft
- [ ] #3 The path is exercised on a profile large enough to reproduce the delay
<!-- AC:END -->
