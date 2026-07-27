---
id: TASK-982
title: Reconcile the Chatbook export directory default across the four windows
status: To Do
assignee: []
created_date: '2026-07-27 19:33'
labels:
  - chatbooks
  - config
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three live Chatbook files default the visible export directory to ~/Documents/Chatbooks while ChatbookCreationWindow.py uses get_private_chatbooks_dir(). Found completing TASK-967 and deliberately not acted on: reconciling in either direction risks orphaning exports a user already has on disk, so the decision needs an owner rather than a sweep. Whichever default wins, the other location needs either a migration or a documented statement that pre-existing exports stay where they are.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The export directory default is the same in all four Chatbook windows,It is decided and written down whether pre-existing exports are migrated or deliberately left,No file composes the path by literal where an accessor exists,A test derives the expected default the way the app does
<!-- AC:END -->
