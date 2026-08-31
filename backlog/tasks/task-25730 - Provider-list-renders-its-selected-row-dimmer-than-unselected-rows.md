---
id: TASK-25730
title: Provider list renders its selected row dimmer than unselected rows
status: To Do
assignee: []
created_date: '2026-08-31 05:10'
labels:
  - console
  - ux-review
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On the first-run provider step the chosen provider is marked with bold and underline, which is a sound non-colour signal, but its text renders at a lower value than its unselected siblings so the selected row reads as the least prominent one. No leading marker is used, which is the cheapest and clearest selection affordance in a terminal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The selected row is at least as prominent as unselected rows
- [ ] #2 Selection is carried by a leading marker in addition to text styling
- [ ] #3 Selection state is apparent without comparing rows side by side
<!-- AC:END -->
