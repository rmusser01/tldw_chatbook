---
id: TASK-32057
title: >-
  Library Collections row: undocumented captures browser, legacy_read_only
  service, and rail side effects
status: To Do
assignee: []
created_date: '2026-09-08 18:23'
labels:
  - library
  - collections
  - docs
  - ux
  - critique-8
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 'Collections' row opens a 'Quick Capture' captures browser ('Sort: saved desc', 'Filter captures', '0–0 of 0') while collections.md still describes create/rename/delete records; LocalLibraryCollectionsService now raises LegacyCollectionsReadOnlyError on every write; the row shows no count until visited, then injects six sub-rows and collapses the Create section, and that collapse persists into the next launch. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 8.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A recorded decision states what the Collections row is today and the user guide matches it
- [ ] #2 The row shows a count before it is visited
- [ ] #3 Selecting the row never changes another rail section's disclosure state
- [ ] #4 If writes are refused, the canvas says so with the recovery path the service names
<!-- AC:END -->
