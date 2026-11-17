---
id: TASK-38936
title: Show agent-created chat lineage and open child conversations
status: To Do
created_date: 2026-11-05 16:08
labels:
- console
- agents
- navigation
priority: medium
references:
- https://uxplanet.org/anthropic-just-shipped-claude-2-0-bbb46a9c1383
- TASK-32482
- TASK-34215
- TASK-34215.2
documentation:
- Docs/Architecture/console.md
- Docs/Architecture/agent-runtime.md
- backlog/decisions/219-console-chat-destinations-and-bounded-starts.md
- backlog/decisions/083-console-edge-rails-and-workspace-tree-ownership.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Help users understand how agent-created conversations relate to the work that started them and open those conversations to inspect or steer them. Keep the difference between independent conversations, message branches, and background subagents understandable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The conversation browser shows parent/child relationships only from authoritative durable chat-creation provenance; legacy, missing, or ambiguous relationships remain explicitly ungrouped or unavailable.
- [ ] #2 Each child remains a real independently openable conversation, with its current status and existing reply/steering controls; background subagents and in-conversation message branches are not represented as equivalent child chats.
- [ ] #3 Grouping preserves workspace ownership, the browser's single ordinary placement of each conversation, existing star/search behavior, and stable keyboard selection.
- [ ] #4 Viewing or expanding lineage neither starts a child nor grants provider, tool, folder, or automatic-work authority.
- [ ] #5 Rename, archive/delete, missing parents, cross-scope relationships, and restart preserve honest reachable identities without reparenting execution or reviving closed work.
- [ ] #6 Mounted keyboard, narrow-layout, and targeted provenance checks cover nested presentation, focus retention, missing parents, and late status updates.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Implementation planning records ADR required: yes/no, ADR path, and reason; any required new or amended ADR is accepted and linked before implementation begins.
- [ ] #2 Applicable targeted automated checks, formatting/lint checks, and live UI or real-provider evidence are recorded before completion; full test sweeps require user opt-in.
- [ ] #3 Relevant user guidance and living Architecture documentation describe the delivered behavior and its limits.
<!-- DOD:END -->
