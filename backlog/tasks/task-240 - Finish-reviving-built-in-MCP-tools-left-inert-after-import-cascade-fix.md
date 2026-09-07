---
id: TASK-240
title: Finish reviving built-in MCP tools left inert after import-cascade fix
status: To Do
assignee: []
created_date: '2026-07-16 15:19'
labels:
  - mcp
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #639 fixed the dead ChaChaNotes_DB import cascade so built-in tools import and mostly execute, but three paths remain inert: chat_with_character hits a NotImplementedError stub (needs a real chat_with_provider equivalent), search_conversations has a residual dead call, and prompts.py:56,125 still call the removed get_conversation_messages. Wire them to real implementations or remove them from the advertised inventory (honesty over stubs).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every tool advertised by get_inventory executes to a real result or a documented graceful error,prompts.py dead calls resolved,chat_with_character either works against a configured provider or is dropped from inventory
<!-- AC:END -->
