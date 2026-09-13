---
id: TASK-1077
title: Validate perform_rag_search inputs like its sibling MCP tools
status: Done
assignee:
  - '@zcode'
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
- [x] #1 perform_rag_search validates its query and limit using Utils/input_validation.py the way search_conversations does,Invalid input returns the tool's normal error dict rather than raising or reaching the backend,A test covers a rejected query and a rejected limit,No other tool in tldw_chatbook/MCP/tools.py accepts an unvalidated query string
<!-- AC:END -->

## Implementation Plan

1. Re-verify AC#4's premise on current dev (tools.py may have grown since filing): inventory every tool taking a query string and its validation status.
2. RED: add tests to Tests/MCP/test_rag_search_tool.py reusing its stub RAG service -- rejected query (too long / blank / non-string) and rejected limit (0 / 101) must return plain error dicts without any backend call; a static AST sweep pins that every query-taking tool in tools.py validates.
3. Apply the exact search_conversations guard (isinstance/strip, validate_text_input, validate_number_range, int coercion) ahead of the try block.

ADR required: no
ADR path: N/A
Reason: Applies the established TASK-985 validation convention to one more handler; no new boundary or policy.

## Implementation Notes

``perform_rag_search`` now validates ``query`` and ``limit`` with the exact guard its sibling ``search_conversations`` established (TASK-985): non-string/blank query, >2000-char query, and limit outside 1..100 each return a single-item ``[{"error": ...}]`` dict before the try block, so invalid input never reaches ``rag_service``. Docstring updated to say so.

TDD evidence: the three new tests were RED on unmodified dev (a 2001-char query reached the stub backend and returned results; the AST sweep flagged perform_rag_search), GREEN after the fix. ``test_valid_inputs_still_reach_the_backend`` pins the happy path (stub records the call; results formatted). AC#4 is pinned permanently by ``test_every_query_taking_tool_validates_its_query``: an AST scan of MCP/tools.py fails if any method with a ``query`` parameter lacks a validate_text_input call -- today that is exactly search_conversations and perform_rag_search, both validated.

Verification: Tests/MCP/test_rag_search_tool.py 8 passed. Full Tests/MCP/ run: 41 failures reproduce identically on stashed HEAD (pre-existing env/optional-deps set; zero new). Ruff: tools.py 16 pre-existing fixables, zero delta; test file formatted clean.

Modified: tldw_chatbook/MCP/tools.py, Tests/MCP/test_rag_search_tool.py.
