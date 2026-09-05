---
id: TASK-31574
title: >-
  Library media Escape - order the strip-vs-Find checks the same way in label
  and action
status: To Do
assignee: []
created_date: '2026-09-05 03:23'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Escape footer label checks the choice strip before the Find bar while the Escape action orders them the other way, so a visible strip with an open Find bar can advertise one thing and do another (Qodo minor on PR #2386, deferred).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Label and action use one ordering
- [ ] #2 A test with a visible strip and an open Find bar asserts the label matches what Escape does
<!-- AC:END -->
