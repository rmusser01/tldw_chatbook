---
id: TASK-524
title: Restore MCP character chat dead dispatch and persistence
status: To Do
assignee: []
created_date: '2026-07-24 00:00'
labels: [mcp, bug]
dependencies: [task-334]
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`MCPTools.chat_with_character` (`MCP/tools.py`) still calls two dead local `NotImplementedError` stubs — `chat_with_provider` (repointable to `chat_api_call`, as TASK-334 did at four other sites) AND `save_conversation_from_messages` (a removed persistence helper with no obvious successor). Restoring MCP character chat needs both: repoint the dispatch AND find/rebuild the conversation-save path. Deferred from TASK-334 because the persistence half needs separate investigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `MCPTools.chat_with_character` dispatches to `chat_api_call(..., streaming=False)` using the same kwarg map as TASK-334 sites
- [ ] #2 MCP character chat performs a real LLM call and returns the response
- [ ] #3 The conversation is persisted (identified and integrated with the conversation-save path, rebuilt if necessary)
- [ ] #4 A test confirms: character chat via MCP works end-to-end and persists to the database
- [ ] #5 The `NotImplementedError` stubs in `MCP/tools.py` are removed once no longer called
<!-- AC:END -->
