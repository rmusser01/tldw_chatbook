---
id: TASK-32231
title: >-
  Library Import queue: identical failed rows are not grouped and there is no
  dismiss-all
status: To Do
assignee: []
created_date: '2026-09-10 14:57'
labels:
  - library
  - import
  - ux
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A folder import with one cause produced four identical `✗ failed` rows × three buttons and no way to clear them at once. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 30.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Identical outcomes group into one row (`✗ failed · 4 files · reason`) with Show the N files / Retry all / Dismiss all, expanding on demand
<!-- AC:END -->
