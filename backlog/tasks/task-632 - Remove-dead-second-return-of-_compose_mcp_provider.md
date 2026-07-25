---
id: TASK-632
title: Remove dead second return element of _compose_mcp_provider
status: To Do
assignee: []
created_date: '2026-07-25'
labels: [tech-debt, tools, tests]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`ConsoleChatController._compose_mcp_provider` (`Chat/console_chat_controller.py:1206`) still returns a 2-tuple `(provider, review_tool_calls)`, building a per-eligible-run `build_mcp_review_hook` closure as its second element. Since TASK-545/P1 introduced the run-level, MCP-independent `build_tool_review_hook`, the real call site (`Chat/console_chat_controller.py:3512`) discards that second element outright: `mcp_provider, _unused_mcp_only_review_hook = await self._compose_mcp_provider()`. The closure is still constructed on every call — wasted work, and a live footgun for the next reader who might assume it is still consumed somewhere.

It is being kept only because two out-of-scope test files pin the 2-tuple shape by unpacking both elements: `Tests/UI/test_console_internals_decomposition.py` (2 call sites) and `Tests/Chat/test_console_agent_swap.py` (6 call sites) — 8 assertions total, all written against `_compose_mcp_provider`'s tuple return before this task's rescope existed and out of scope for TASK-545 itself. Clean this up once those tests can be touched: either drop the second element and update all 8 call sites to unpack a single value (or a small named result), or — if the closure genuinely still has a legitimate independent consumer — document why it must stay and drop the `_unused_` naming that currently signals dead weight.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] `_compose_mcp_provider` returns only what is actually consumed by its production call site, or its second element's continued existence is justified in its docstring
- [ ] All test call sites (`Tests/UI/test_console_internals_decomposition.py`, `Tests/Chat/test_console_agent_swap.py`) are updated to match the (possibly narrowed) return shape and pass
- [ ] No behavior change to the composed `MCPToolProvider` or to MCP tool-call review/approval flow
<!-- AC:END -->
