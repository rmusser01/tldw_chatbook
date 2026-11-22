---
id: TASK-38936
title: Show agent-created chat lineage and open child conversations
status: To Do
created_date: 2026-11-05 16:08
updated_date: 2026-11-22 15:51
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
- backlog/decisions/348-agent-chat-fork-and-spawn.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Help users understand how agent-created conversations relate to the work that started them and open those conversations to inspect or steer them. Keep the difference between independent conversations, message branches, and background subagents understandable.

The durable provenance is what ADR-348's tools write (`Chat/console_chat_controller.py`, `execute_agent_chat_create` and the prepared new-chat path): `fork_chat` writes the FK-enforced `parent_conversation_id` and `forked_from_message_id` columns, while `new_chat` writes only `console_agent_handoff` metadata (`created_via`, `source_run_id`); the AgentRuns row for that run records the conversation it ran in. Both tools stamp `console_agent_handoff.created_via`. A user fork writes the same lineage columns without that stamp.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The conversation browser shows parent/child relationships only from authoritative durable chat-creation provenance; legacy, missing, or ambiguous relationships remain explicitly ungrouped or unavailable.
- [ ] #2 Each child remains a real independently openable conversation, with its current status and existing reply/steering controls; background subagents and in-conversation message branches are not represented as equivalent child chats.
- [ ] #3 Grouping preserves workspace ownership, the browser's single ordinary placement of each conversation, existing star/search behavior, and stable keyboard selection.
- [ ] #4 Viewing or expanding lineage neither starts a child nor grants provider, tool, folder, or automatic-work authority.
- [ ] #5 A `fork_chat` child is grouped under the conversation named by its `parent_conversation_id`; a `new_chat` child is grouped under the conversation of its `source_run_id` run when that AgentRuns row exists, and is shown ungrouped otherwise; a user fork (lineage columns without a `console_agent_handoff` stamp) is not shown as an agent-created child.
- [ ] #6 Renaming a parent or child changes only its displayed title; an archived, deleted, or unreadable parent leaves its children listed and openable, marked as having an unavailable parent, and never grouped under another conversation; a child in a different workspace or scope from its parent is listed only in its own, with the relationship labeled; after restart the same grouping is rebuilt from persisted provenance alone; opening or expanding a child never starts, resumes, or re-runs a run or a pending draft launch.
- [ ] #7 Mounted keyboard, narrow-layout, and targeted provenance checks cover both provenance sources, nested presentation, focus retention, rename, missing parents, cross-workspace children, restart, and late status updates.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Implementation planning records ADR required: yes/no, ADR path, and reason; any required new or amended ADR is accepted and linked before implementation begins.
- [ ] #2 Automated tests (unit + integration) cover any new logic, the linter and formatter pass, and `./scripts/preflight.sh` passes, per CLAUDE.md's Definition of Done.
- [ ] #3 Relevant user guidance and living Architecture documentation describe the delivered behavior and its limits.
<!-- DOD:END -->
