---
id: TASK-31235
title: Sort chooser renders every option
status: To Do
assignee: []
created_date: '2026-09-04 01:50'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P1: the sort chooser renders "✓ Newest  Oldest  Title A-" in the ~38-col items pane — "Title A-Z" is truncated and "Title Z-A" is entirely invisible with no overflow cue, yet a keyboard user can still arrow to and select the unrendered option. Recognition-over-recall failure: an option you can't see doesn't exist. Verified: the sort chooser is a horizontal compose_library_choice_strip (library_media_canvas.py:690-701) while the sibling type chooser is a vertical OptionList (line 681) that fits fine.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All four MEDIA_SORT_CHOICES options are fully visible when the sort chooser is open at the shell's narrow pane width
- [ ] #2 The active option keeps its ✓ marker and keyboard selection works over the visible list
- [ ] #3 A test pins that every sort option's label renders (painted-text or equivalent, not just presence in the DOM)
<!-- AC:END -->
