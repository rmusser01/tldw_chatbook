---
id: TASK-239
title: MCP Hub Tools-mode polish batch from Phase 3 reviews
status: To Do
assignee: []
created_date: '2026-07-16 15:19'
labels:
  - mcp
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Small UX/observability items from PR #639 reviews, none blocking: (1) 'Phase 4' leaks into user copy — use 'a later phase' like siblings; (2) timeout durations render as 45012ms — switch to seconds above ~1s; (3) in-flight toast shows the raw packed tool_id — show the tool name; (4) integer coercion error says 'must be a number' — 'must be an integer'; (5) server-source empty state routes connect/refresh to Servers mode where those actions are disabled for that source — adjust copy or routing; (6) silent duplicate-row skip in _apply_filter needs a logger.debug; (7) filter-bar/#mcp-inspector-tool geometry lives only in DEFAULT_CSS — add bundle lockstep rules per convention.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All seven items addressed or explicitly declined with reasons in this task,Copy changes covered by the existing exact-text tests
<!-- AC:END -->
