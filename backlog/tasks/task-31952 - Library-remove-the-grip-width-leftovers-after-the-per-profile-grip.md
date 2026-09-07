---
id: TASK-31952
title: Library - remove the grip-width leftovers after the per-profile grip
status: To Do
assignee: []
created_date: '2026-09-07 08:26'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
H final review: every LibraryAdaptiveReaderPaneGrip now sets its width inline from the profile, so .library-adaptive-reader-pane-grip { width/min-width/max-width: 5 } in _agentic_terminal.tcss (~1281) is dead for all destinations, not only Media. library_screen.py:104 imports PANE_GRIP_WIDTH unused, and library_skills_controller.py:806 computes 2 * PANE_GRIP_WIDTH where 2 * LIBRARY_SKILLS_READER_PROFILE.grip_width is the honest expression - correct today, wrong the moment a second profile opts in. Carrying grip_width on AdaptiveReaderEffectiveLayout would make the resolver's reservation and the shell's paint one number (seven construction sites).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The dead grip width rule is gone from the component source and from the rebuilt bundle
- [ ] #2 No module reads PANE_GRIP_WIDTH where a profile's grip_width is the real number
- [ ] #3 The width the resolver reserves and the width the shell paints come from one value
<!-- AC:END -->
