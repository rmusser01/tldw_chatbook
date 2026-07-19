---
id: TASK-294
title: MCP approval + audit UX polish from Phase 5 reviews
status: To Do
assignee: []
created_date: '2026-07-17 19:18'
labels:
  - mcp
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
sr-UX-adjacent polish deferred from PR #675: approval card shows no countdown/deadline hint (silently vanishes at 120s); 'Approve all' requires a second Submit click; the deny refusal copy 'blocked by MCP permissions (set to Off)' is reused for explicit user denials (misleading provenance); Audit 'When' column renders raw UTC ISO without tz conversion; test docstring says height<=3 while asserting <=4. Also fold the remaining P5 test-hygiene minors: hook test against spawn/find/load names, ChatTaskCards.sync_state batch branch direct test, misnamed collapse test, tool_naming dedupe docstring warning.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each item addressed or explicitly declined with reasons in this task
<!-- AC:END -->
