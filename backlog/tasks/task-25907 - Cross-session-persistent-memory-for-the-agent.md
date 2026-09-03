---
id: TASK-25907
title: Cross-session persistent memory for the agent
status: To Do
assignee: []
created_date: '2026-08-31 15:09'
updated_date: '2026-09-02 06:50'
labels:
  - agents
  - memory
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Chatbook's memory is strictly per-conversation: Chat/console_context_repository.py:136 stores compaction summaries scoped to a conversation and its lineage, and a named grep for MEMORY.md, persistent memory, memory_store and user_profile across Agents/ and Chat/ returns zero. The only learning path is Agents/agent_lesson_promotion.py, which is human-gated and writes into AGENTS.md. Hermes carries MEMORY.md plus USER.md, a MemoryProvider interface with nine external backends, a scheduled curator that ages and consolidates learned facts, and a journey graph. This is the largest build in the 2026-08-31 parity report's top ten and is filed as a foundation task: it needs a design pass before implementation, and the privacy boundary (what may persist across conversations, and whether it ever leaves the machine) is the decision that matters most.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A design is recorded as an ADR covering: what may be written, who writes it, when it is read, where it is stored, and the retention and deletion story
- [ ] #2 The ADR states the privacy boundary explicitly, consistent with the local-private-data stance, and whether any of it is ever synced
- [ ] #3 The ADR states how persisted memory interacts with the existing approval-gated lesson promotion rather than duplicating it
- [ ] #4 A first implementation slice is scoped in the ADR as an independently shippable follow-up, not built under this task
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/<next> - cross-session-agent-memory.md. Reason: this creates a new durable store of user-derived data and a new automatic write path; the privacy and retention decisions are the user's, not the implementer's. Sweep the decisions directory for a free number at authoring time - ADR numbers in this repo collide routinely.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DEFERRED by owner (2026-09-02) until Personal Context 02 (interviews + agent tools) lands. Re-scope recorded: chatbook already has three memory pillars — Notes (freeform), Agent-Lessons (human-gated learned facts -> AGENTS.md), and the in-flight Personal Context program (encrypted USER-profile core, plans 01-04 merged, append_personal_context live in agent_service). Any 25907 ADR should be a short POSITIONING decision over those pillars (no fourth store), mapping the four deferred rows onto them (curator -> later lessons/notes maintenance automation; external MemoryProviders + journey graph -> rejected or server-side). Do not design a parallel memory system.
<!-- SECTION:NOTES:END -->
