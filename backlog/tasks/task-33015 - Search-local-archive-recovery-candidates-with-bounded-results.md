---
id: TASK-33015
title: Search local archive recovery candidates with bounded results
status: To Do
assignee: []
created_date: '2026-09-27 15:22'
labels: []
dependencies:
  - TASK-33014
documentation:
  - Docs/superpowers/plans/2026-09-27-agent-archive-recovery.md
  - backlog/decisions/166-agent-assisted-archive-recovery.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Find missing archived chats across local workspaces without an embedding index.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Search includes chat and workspace-only archives while excluding deleted records and private control content.
- [ ] #2 Either applicable archive event can match date filters; unknown times and live continuation are explicit.
- [ ] #3 Storage work and output are bounded; stale or incomplete queries never return false complete results.
<!-- AC:END -->
