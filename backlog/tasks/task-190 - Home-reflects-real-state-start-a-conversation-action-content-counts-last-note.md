---
id: TASK-190
title: >-
  Home reflects real state: start-a-conversation action, content counts, last
  note
status: To Do
assignee: []
created_date: '2026-07-12 03:05'
labels:
  - ux
  - home
  - enhancement
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Upgrade opportunity from core-loop UAT 2026-07-11: Home's canvas shows only an Import Library sources card even when the provider is ready and conversations/notes exist. Home should mirror actual state: provider ready -> a primary Start a conversation action; counts for conversations/notes/media; most recent note or conversation as a resume entry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When the provider is ready Home offers a primary start-a-conversation control that routes to Console,Home shows real content counts sourced from the same seams the Library rail uses,A most-recent item (note or conversation) is surfaced as a one-click resume,Empty profile still gets the import suggestion
<!-- AC:END -->
