---
id: TASK-38401
title: >-
  Console: the longer Temporary segment pushes the run segment off a narrow
  status strip
status: To Do
assignee: []
created_date: '2026-10-31 01:38'
labels:
  - console
  - copy
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-37840's owner-specified copy grew the status strip's Temporary segment from 22 cells ('Temporary · traces off') to 34 ('Temporary · not saved · traces off' or '... · capture on'). The run segment ('Run: Blocked — save chat first', about 30 cells) was placed left-most so it stays visible when the strip scrolls, but in a Temporary chat it now starts near column 38. At 80x24 both still fit; below about 70 columns the run segment scrolls off, so a paused send's reason is no longer on screen. Raised by the pre-merge review of TASK-37840. Whether to add a compact form under a width threshold, reorder the segments, or leave it is a UX and owner call, because the copy is owner-specified.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision on the Temporary segment's narrow-width form is recorded on this task
- [ ] #2 At the narrowest supported Console width, a Temporary chat whose send is paused shows the run segment's reason on screen without scrolling the strip, or this task records why it should not
- [ ] #3 A test pins the narrow-width rendering the decision requires
<!-- AC:END -->
