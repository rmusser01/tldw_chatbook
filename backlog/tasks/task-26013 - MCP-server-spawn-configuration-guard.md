---
id: TASK-26013
title: MCP server-spawn configuration guard
status: To Do
assignee: []
created_date: '2026-08-31 15:44'
labels:
  - security
  - mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Any command can be saved as an MCP server and spawned. Verified on origin/dev: MCP/local_store.py sanitizes environment values and rejects secret-shaped literals (:42-50), but a named grep for authorized_keys, IOC and suspicious across tldw_chatbook/MCP returns zero - nothing inspects the command itself. A pasted server config is a code-execution primitive, and imported configs (MCP/mcp_import.py:51 reads Claude Desktop JSON) are exactly the untrusted-input path. Hermes blocks shell-interpreter egress and persistence shapes plus known indicators at both save time and spawn time.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A server configuration whose command matches a known dangerous shape is refused at save time with the reason stated
- [ ] #2 The same check runs at spawn time, so a configuration edited on disk cannot bypass the save-time check
- [ ] #3 Imported configurations pass through the identical check as hand-entered ones
- [ ] #4 Shapes covered include at minimum: piping a remote fetch into an interpreter, writing to shell startup files or authorized_keys, and inline interpreter invocation of encoded payloads
- [ ] #5 A refusal names the matched rule and does not silently drop the server from the list
- [ ] #6 Ordinary server configurations (npx, uvx, python -m, a plain binary path) are unaffected, asserted by tests
<!-- AC:END -->
