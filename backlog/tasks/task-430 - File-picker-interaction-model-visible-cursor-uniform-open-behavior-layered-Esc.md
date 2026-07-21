---
id: TASK-430
title: File picker interaction model - visible cursor, uniform open behavior, layered Esc
status: To Do
assignee: []
created_date: '2026-07-21 09:38'
labels:
  - widgets
  - ux
  - file-picker
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Observed live while importing a card (EnhancedFileOpen): the selection cursor is bold+underline with a near-identical background and is effectively invisible; single-click on a directory navigates immediately while files require select+Enter (clicking ".." just to focus the list teleports you up a level); entering a full FILE path in the Ctrl+L bar then Enter/Go only navigates to the parent listing instead of opening the file; Esc pressed inside the Recent overlay dismisses the entire picker. An expert driver needed ~10 interactions to open one file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Selected row is clearly distinguishable at a glance in the default theme
- [ ] #2 Directories and files share one predictable activation model (single-click selects for both; a consistent action opens/descends)
- [ ] #3 Confirming a full file path in the path bar opens/returns that file
- [ ] #4 Esc dismisses only the topmost overlay (Recent/Bookmarks), not the whole picker
<!-- AC:END -->
