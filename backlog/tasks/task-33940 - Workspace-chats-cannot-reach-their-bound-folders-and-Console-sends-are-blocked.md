---
id: TASK-33940
title: >-
  Workspace chats cannot reach their bound folders, and Console sends are blocked (live UAT 2026-10-02)
status: To Do
assignee: []
created_date: '2026-10-02 20:56'
labels:
  - workspace
  - console
  - uat-2026-10-02
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users report that a chat in a workspace does not see the workspace's own files and only knows about its private scratch space and the folder the app was launched from. A live investigation on dev 2d34cbf80d (real app, isolated profile, real OpenAI model, workspace and chat created through the UI) confirmed it and found three more defects on the same path, each of which stops or degrades a real send. The owner asked for all of them to be fixed in one PR. Each subtask carries its own evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every subtask is Done and its live evidence is recorded in its Implementation Notes
- [ ] #2 A new chat in a workspace whose folder lies outside the launch directory reads that folder's files on its first tool call, in the real app with a real model
<!-- AC:END -->
