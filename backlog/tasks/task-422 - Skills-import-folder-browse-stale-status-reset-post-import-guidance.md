---
id: TASK-422
title: 'Skills import - folder browse, stale status reset, post-import guidance'
status: To Do
assignee: []
created_date: '2026-07-21 15:19'
labels:
  - skills
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2 from the 2026-07-21 Skills UX/NNG review (stale status verified live). Import row friction: (1) Browse opens a file-only picker while the placeholder invites a skill FOLDER path - the common one-directory-per-skill shape must be typed by hand; (2) the open import row and its last status/error persist across editor round-trips and resurface stale minutes later; (3) success copy says 're-review it in the trust panel' with no way to get there from the message. NNG heuristics 7 and 1.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Importing a skill directory is possible without typing the path by hand,Returning to the list view resets or closes the import row so stale errors do not resurface,Successful import gives a direct path to the imported skill's trust panel (open row or equivalent),Covered by tests
<!-- AC:END -->
