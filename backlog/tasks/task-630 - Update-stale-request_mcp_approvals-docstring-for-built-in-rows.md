---
id: TASK-630
title: Update stale request_mcp_approvals docstring for built-in tool rows
status: To Do
assignee: []
created_date: '2026-07-25'
labels: [docs, tools, tech-debt]
dependencies: [TASK-545]
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`ConsoleChatController.request_mcp_approvals` (`Chat/console_chat_controller.py:974`) still opens with the docstring "Bridge one batch of pending MCP tool calls to the Console UI and back" and describes itself as "Bound as `MCPToolProvider`'s `approval_callback`". Since TASK-545/P1's run-level `build_tool_review_hook`, this method is owner-agnostic: it is handed the same batch of pending calls regardless of whether each row originated from an MCP tool or a built-in agent-runtime tool (`server_key="agent:builtin"`), and it marshals both to the same `ChatApprovalCard` and back through the same `Event`-polling loop. The name and the docstring's framing both predate that change and now describe only half of what the method does, which will mislead the next reader into assuming built-in approvals go through some other path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] `request_mcp_approvals`'s docstring no longer claims the batch is exclusively "MCP tool calls" or that the method is bound only as `MCPToolProvider`'s `approval_callback`
- [ ] The docstring states plainly that the method is owner-agnostic and serves both MCP and built-in (`agent:builtin`) approval rows
- [ ] No behavior change — this is a documentation-only fix, verified by an unchanged diff outside the docstring/comment
<!-- AC:END -->
