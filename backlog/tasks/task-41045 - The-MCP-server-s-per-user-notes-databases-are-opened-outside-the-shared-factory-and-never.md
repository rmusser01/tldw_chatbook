---
id: TASK-41045
title: "The MCP server's per-user notes databases are opened outside the shared factory and never closed"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- architecture
- mcp
- database
priority: low
references:
- tldw_chatbook/Notes/Notes_Library.py
- tldw_chatbook/MCP/server.py
- tldw_chatbook/app_services.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`NotesInteropService` (`Notes/Notes_Library.py:179`) lazily opens one `CharactersRAGDB` per user_id. In the MCP server, these handles bypass the profile-storage factory that TASK-36947.7 introduced. Nothing calls `close_all_user_connections()` at shutdown either (no code calls it at all), so the handles are left to process exit with no WAL checkpoint.

TASK-36947.7's review points out that a plain close is not enough: shutdown needs a quiesce-style close (`quiesce_connections`) that closes every registered handle on the same file. Re-read on dev ecad404bd4 (2026-11-15).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The MCP server opens per-user notes handles through a factory opener
- [ ] #2 MCP shutdown closes every per-user handle with a quiesce-style close
- [ ] #3 A test shows that no handle is left open after the MCP server shuts down
<!-- AC:END -->
