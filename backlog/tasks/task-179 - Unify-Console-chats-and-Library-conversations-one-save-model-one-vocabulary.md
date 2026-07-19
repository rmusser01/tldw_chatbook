---
id: TASK-179
title: 'Unify Console chats and Library conversations (one save model, one vocabulary)'
status: Done
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
- [x] #1 A conversation created by chatting in Console is discoverable from Library Browse Conversations (or the rail copy stops calling unsaved chats saved)
- [x] #2 Console and Library use the same term for the same persistence state
- [x] #3 Library conversation count matches what Console lists for the same profile
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed on branch claude/uat-core-loop-2026-07 (PR #606, commits 6fd4a60f..88c0475b) with focused tests; re-verified live against llama.cpp on a fresh profile (remediation captures in Docs/superpowers/qa/core-loop-uat-2026-07).
<!-- SECTION:NOTES:END -->
