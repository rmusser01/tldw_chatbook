---
id: TASK-38410
title: >-
  Console: no keyboard route expands a collapsed token where the terminal keeps
  Ctrl+E
status: To Do
assignee: []
created_date: '2026-10-31 02:37'
labels:
  - console
  - composer
  - ux-review-2026-09-29
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 2026-10-30 owner decision made Ctrl+E the only key that expands a collapsed paste, file or /prompt token (Enter always sends). Some terminals never pass Ctrl+E to the app: VS Code's integrated terminal on Windows and Linux binds it to Quick Open and keeps it from the shell by default. There a keyboard-only user cannot expand a token at all; only a click works. The User Guide now says so. Found in the TASK-37870 review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A keyboard-only user can expand a collapsed token without Ctrl+E (for example a command-palette command or a composer menu item), pinned by a Pilot test on the real ChatScreen that fails before the change
- [ ] #2 Docs/User_Guide/console/chat-basics.md 'Large pastes' names that route next to the Ctrl+E caveat
<!-- AC:END -->
