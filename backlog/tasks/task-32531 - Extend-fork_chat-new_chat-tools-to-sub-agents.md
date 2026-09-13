---
id: TASK-32531
title: Extend fork_chat/new_chat tools to sub-agents
status: To Do
assignee: []
created_date: '2026-09-12 01:17'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
v1 pins fork_chat/new_chat runtime tools to primary agents only. Sub-agents (fleet children, skill spawns) are scoped to the parent conversation, so the same tools can let a delegated child open follow-up workstream chats for the user with the same confirmation gates. Build on the v1 spec: Docs/superpowers/specs/2026-09-11-agent-chat-fork-spawn-design.md (see Sub-agent extension section).
<!-- SECTION:DESCRIPTION:END -->

## Renumbering provenance

Renumbered from TASK-32480 during PR #2649 rebase on 2026-09-13: the ID
collides with dev's older "Tokenize features/_chat.tcss" task, and the
duplicate-backlog-ID CI gate fails on PR-touched collisions. TASK-32531 is
free across fetched remote refs and the local worktree.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Sub-agent runs advertise fork_chat and new_chat schemas when injected
- [ ] #2 Fork from a sub-agent copies the parent conversation's active path with lineage columns set
- [ ] #3 Confirmation card identifies the requesting agent and parent run
- [ ] #4 Denial and session-remember semantics scope to the requesting run
- [ ] #5 Fleet-child fork and new_chat paths tested end to end
<!-- AC:END -->
