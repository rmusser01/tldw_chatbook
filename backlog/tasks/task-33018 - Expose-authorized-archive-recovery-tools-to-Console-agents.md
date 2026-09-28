---
id: TASK-33018
title: Expose authorized archive recovery tools to Console agents
status: To Do
assignee: []
created_date: '2026-09-27 15:26'
labels: []
dependencies:
  - TASK-33015
  - TASK-33017
documentation:
  - Docs/superpowers/plans/2026-09-27-agent-archive-recovery.md
  - backlog/decisions/166-agent-assisted-archive-recovery.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make recovery available under either Library retrieval mode with the approved authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reserved recovery tools coexist with Direct or RAG through authenticated runtime registration.
- [ ] #2 Only eligible interactive primary runs with Library access execute; blocked, unattended, temporary and subagent paths refuse.
- [ ] #3 Production dispatch preserves execution gates and does not expose recovery through shared MCP descriptors.
<!-- AC:END -->

## Filing provenance

Created with Backlog CLI in a temporary Git workspace using the project configuration, after the production CLI repeatedly scanned local branch history. Copied into this project with a unique ID above the refreshed all-ref/worktree ceiling and current local task IDs; dependencies and contents were verified.
