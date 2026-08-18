---
id: TASK-17655
title: 'Console: design pass — inline REPL prompt (north star)'
status: To Do
assignee: []
created_date: '2026-08-17'
labels:
  - console
  - ux
  - design
dependencies:
  - task-17651
  - task-17652
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design-only task, Phase C of the 2026-08-17 bottom-stack de-clutter: explore moving the prompt inline into the transcript flow, toad-style (reference: batrachianai/toad, whose MainScreen composes a Conversation widget plus a single Footer; the prompt lives in the conversation with a mode glyph and a hidden-when-empty status line, and every chrome bar is an individual ui.* visibility setting).

This is a structural change that must find a new home for every current composer affordance — menu, attachments, paste-collapse, recovery states, dictation, prompt queue, staged evidence — and either honor or explicitly revise the owner's keep-all-send-affordances ruling. The output is an approved spec, not code; implementation tasks are filed only after the owner approves the spec.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A spec documents the inline-prompt architecture, the new home of every current composer affordance, the interaction model (send, queue, run status, approvals), and the migration and testing strategy
- [ ] #2 The owner has explicitly approved the spec before any implementation task is filed
- [ ] #3 Implementation tasks exist only after approval and reference the approved spec
<!-- AC:END -->
