---
id: TASK-1077
title: Validate perform_rag_search inputs like its sibling MCP tools
status: To Do
assignee: []
created_date: '2026-07-27 21:47'
labels:
  - mcp
  - security
  - validation
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
perform_rag_search in tldw_chatbook/MCP/tools.py accepts a query string and a limit from an MCP caller and passes both straight through to the search backends without validation. It is the only tool in that file that takes a query string and does not validate it -- verified by parsing every async def in the module and checking each body for validate_text_input. Its sibling search_conversations was brought in line during TASK-985 (PR #1024), which established the pattern and the import; this one was deliberately left alone at the time rather than silently extending a convention change beyond that task's scope. CLAUDE.md states inputs are validated at boundaries, and an MCP tool handler is a boundary: the caller is a model, and on a hub-connected setup the values can originate outside the user's own machine.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 perform_rag_search validates its query and limit using Utils/input_validation.py the way search_conversations does,Invalid input returns the tool's normal error dict rather than raising or reaching the backend,A test covers a rejected query and a rejected limit,No other tool in tldw_chatbook/MCP/tools.py accepts an unvalidated query string
<!-- AC:END -->
