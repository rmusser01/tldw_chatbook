---
id: TASK-32642
title: >-
  Library Notes: Info tab's blank rows become a two-column properties layout
status: To Do
assignee: []
created_date: '2026-09-15 10:35'
labels:
  - library
  - notes
  - critique-4
  - idea
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 idea 8's unlanded half, ACCEPTED in task-32627. task-32143 already
delivered the editor chrome strip (word count, cursor line, save state, the
main actions without tabbing). What it did not do is Info: roughly 25 blank
rows below a short single-column list.

A two-column properties layout fills that, and keywords moving inline under
the title is the other half of the same idea — both are density, not new
information.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Info uses its height: no run of blank rows below the last property at 235x52.
- [ ] #2 The layout collapses to one column at 100x30 rather than truncating values.
- [ ] #3 Keywords are reachable and editable from the editor without opening Info.
- [ ] #4 No property is removed to make the layout fit — the fix is arrangement, not loss.
<!-- AC:END -->
