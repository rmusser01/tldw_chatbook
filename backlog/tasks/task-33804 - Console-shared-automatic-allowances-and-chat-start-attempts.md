---
id: TASK-33804
title: Console shared automatic allowances and chat-start attempts
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-02 23:24'
updated_date: '2026-10-02 23:35'
labels:
  - console
  - agents
dependencies:
  - TASK-33802
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep agent-created chats within the originating automatic-work allowance while preserving each conversation's run ownership and durable execution fences.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Existing and migrated manual chains retain their finite limits and conversation ownership; automatic descendants refer directly to an immutable canonical allowance root.
- [ ] #2 Counters, deadlines, pause/review state, settlement uncertainty, and recovery apply across every member of a shared allowance, including concurrent reservations and late callbacks.
- [ ] #3 Durable exact-target chat-start attempts prepare, accept once, abort only before acceptance, and retain accepted or uncertain charges without replay after restart.
- [ ] #4 Automatic execution context validates native chat-start attempts and refuses stale runtime owners without fabricating wake claims or cross-conversation run parents.
- [ ] #5 Targeted SQLite, migration, race, deadline, and automatic-context checks pass with the existing manual and fleet-wake behavior preserved.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/211-console-chat-destinations-and-bounded-starts.md
Reason: Implements the approved shared automatic-allowance ownership and durable chat-start authority contract.

Planning phase only; no application implementation or acceptance criterion has been completed.
Detailed plan: Docs/superpowers/plans/2026-10-02-console-chat-destinations-and-starts.md, Task 1.
Spec: Docs/superpowers/specs/2026-10-02-console-chat-destinations-and-starts-design.md.

1. Capture targeted baseline, current schema and formatter evidence.
2. Add real-SQLite failing shared-allowance and native-attempt tests.
3. Implement AgentRunsDB migration, direct immutable root membership and root-wide admission/settlement/recovery.
4. Implement native start-attempt preparation/acceptance/abort/completion and accepted automatic context, preserving run conversation scope.
5. Verify races, deadlines, migration, uncertain accounting and existing wake/manual behavior; update implementation notes only after completion.

Allocation provenance: CLI assigned TASK-33803. A fresh scan of all locally available Git refs and 114 worktrees found IDs through 33803; this task was renamed to TASK-33804 before references were added.
<!-- SECTION:PLAN:END -->
