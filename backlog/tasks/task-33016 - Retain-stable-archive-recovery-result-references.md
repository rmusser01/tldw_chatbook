---
id: TASK-33016
title: Retain stable archive recovery result references
status: To Do
assignee: []
created_date: '2026-09-27 15:26'
labels: []
dependencies:
  - TASK-33015
documentation:
  - Docs/superpowers/plans/2026-09-27-agent-archive-recovery.md
  - backlog/decisions/166-agent-assisted-archive-recovery.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep displayed recovery choices attached to exact identities across searches and retries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Session-owned result references enforce the approved retention and expiry limits.
- [ ] #2 Wrong-owner, expired, changed-argument and repeated calls cannot acquire restoration authority.
- [ ] #3 Manual refresh and imported transcript content cannot create model evidence or trusted controls.
<!-- AC:END -->

## Filing provenance

Created with Backlog CLI in a temporary Git workspace using the project configuration, after the production CLI repeatedly scanned local branch history. Copied into this project with a unique ID above the refreshed all-ref/worktree ceiling and current local task IDs; dependencies and contents were verified.
