---
id: TASK-32388
title: 'Library Conversations: after a workspace hop, an Undo receipt names the wrong workspace'
status: To Do
assignee: []
created_date: '2026-09-11 10:30'
labels:
  - library
  - conversations
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-32107 made "Use as source" link a conversation into the active workspace in one undoable step, with a receipt and an "Undo link" beside it, and made Undo remove the membership the receipt names rather than whichever workspace happens to be active. A residual remains: switch to a second workspace that also holds that membership and the receipt on screen still names the first one, so Undo correctly removes the membership the receipt names while the user reads it as acting on the workspace they are now in. Recorded in task-32107's own notes at close.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Switching the active workspace stands the stale link receipt down, or restates it for the workspace now active
- [ ] #2 Undo never removes a membership the receipt on screen does not name
- [ ] #3 The workspace-hop case is covered by a test alongside the existing link/undo pins
<!-- AC:END -->
