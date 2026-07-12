---
id: TASK-179
title: 'Unify Console chats and Library conversations (one save model, one vocabulary)'
status: To Do
assignee: []
created_date: '2026-07-12 02:47'
labels:
  - ux
  - library
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Core-loop UAT 2026-07-11: Console rail labels its auto-persisted chats 'saved workspace - Nm' while Library shows Conversations (0) with 'No saved conversations yet. Save a Console chat and it appears here.' DB-seeded conversations appear in both surfaces. Two contradictory definitions of a saved conversation read as data loss to the user.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A conversation created by chatting in Console is discoverable from Library Browse Conversations (or the rail copy stops calling unsaved chats saved)
- [ ] #2 Console and Library use the same term for the same persistence state
- [ ] #3 Library conversation count matches what Console lists for the same profile
<!-- AC:END -->
