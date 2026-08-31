---
id: TASK-25723
title: Composer actions menu spends thirty rows on six items
status: To Do
assignee: []
created_date: '2026-08-31 05:09'
labels:
  - console
  - ux-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The composer menu spreads six actions over roughly thirty rows with items centred rather than left aligned, anchors itself at the top left while its trigger sits at the bottom of the screen, and shows no keyboard accelerators. Reasons for disabled items are present and well written but separated from their action by blank rows, weakening the association. The result occludes the transcript to present very little.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Menu items are left aligned and vertically compact enough to scan without scrolling
- [ ] #2 The menu is anchored adjacent to the control that opens it
- [ ] #3 A disabled item's reason is visually bound to that item
- [ ] #4 Keyboard accelerators are shown for actions that have them
<!-- AC:END -->
