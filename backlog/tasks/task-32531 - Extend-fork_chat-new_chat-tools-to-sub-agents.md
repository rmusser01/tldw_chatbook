---
id: TASK-32531
title: Extend fork_chat/new_chat tools to sub-agents
status: Done
assignee:
  - '@robert'
created_date: '2026-09-12 01:17'
updated_date: '2026-09-20 19:29'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Lift the primary-only gates for chat-create only: plan-builder kind gate, _chat_create_runtime_schemas, LoopDeps population (install_skill keeps its primary gate). 2. Card enrichment stamps requesting-run identity (agent_kind, parent_run_id, task snippet) from the agent_runs row; card renders 'requested by sub-agent' line. 3. Session-grant short-circuit skipped when the requesting run is a sub-agent (per-call confirm always for children); denial counter unchanged (per-closure). 4. Tests: subagent-kind disclosure+population, enrichment identity, grant skip, end-to-end fork from a child-run context.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Merged (see PR): primary gates lifted for chat-create; card identity for sub-agent requesters; grants never ride for children; end-to-end child-context fork tested.
<!-- SECTION:NOTES:END -->
