---
id: TASK-32661
title: 'File picker: the listing shows one content row at 100x30'
status: To Do
assignee: []
created_date: '2026-09-15 23:25'
labels:
  - library
  - picker
  - ux
  - layout
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In the vendored file picker the listing pane resolves to a single visible content row on a 100x30 terminal, so choosing between folders means scrolling one row at a time through a list whose whole purpose is comparison. Pre-existing, not caused by any wave-5 change; found while writing a compositor assertion for task-32643's folder badge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 100x30 the picker's listing shows several content rows, enough to compare folders without scrolling one row at a time
- [ ] #2 The chrome that is traded away for those rows is named, and the trade is deliberate rather than whatever `height: 1fr` resolves to
- [ ] #3 A pin reads the visible row count off the compositor at 100x30, so a future chrome addition cannot silently take the listing back down to one row
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Measured during task-32643: the listing is `height: 1fr` between the title, the path field, the filter, the sort controls, the hint line, the button row and the status line, and at 30 rows that leaves it one. A paint assertion for the folder badge had to be moved to 235x52 to mean anything, which is what surfaced it.

Note for whoever takes it: task-32643's per-row note badge makes this worse in KIND, not in code -- a one-row listing is exactly where a per-row badge buys least. Fixing the rows is the fix; the badge needs no change.
<!-- SECTION:NOTES:END -->
