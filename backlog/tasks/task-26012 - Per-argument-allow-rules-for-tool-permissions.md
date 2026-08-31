---
id: TASK-26012
title: Per-argument allow rules for tool permissions
status: To Do
assignee: []
created_date: '2026-08-31 15:44'
labels:
  - security
  - mcp
dependencies:
  - TASK-25905
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Always-allow is all-or-nothing per tool, so safe repeats can never be quieted. Verified on origin/dev: MCP/permission_store.py:472,489 keys state by (server_key, tool_name) with no argument dimension - which is precisely why the approval card deliberately withholds always_allow for raw shell (Widgets/Chat_Widgets/chat_approval_card.py:57), since allowing shell_exec once would allow every command. The result is a real capability gap wearing a safety justification: a user approving the same harmless command twenty times has no way to stop being asked. Hermes scopes allow rules to command-text globs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An always-allow entry can be scoped to an argument predicate rather than the whole tool
- [ ] #2 A call matching the predicate is allowed; the same tool with non-matching arguments still prompts
- [ ] #3 Predicates are displayed to the user in full before they are saved - no rule is created from a call the user did not read
- [ ] #4 Argument-scoped rules participate in the existing definition-hash rug-pull guard: a changed tool definition invalidates them
- [ ] #5 High-risk tools remain floored to ask regardless of an argument rule, consistent with MCP/permission_store.py:912-918
- [ ] #6 Raw shell can adopt argument-scoped allow only in combination with the hardline floor from task-25905 - stated explicitly in the notes if not implemented here
<!-- AC:END -->
