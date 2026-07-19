---
id: TASK-237
title: Record CancelledError and pre-flight rejections in the MCP execution log
status: To Do
assignee: []
created_date: '2026-07-16 15:19'
labels:
  - mcp
  - phase5
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
test_hub_tool records success/failure/timeout but not cancellations (sibling _run_local_lifecycle does) and raises unknown-prefix ValueError before the recording try, so those attempts never reach the log. Audit mode (Phase 5) reads these records; decide and implement consistent every-attempt semantics (PR #639 Task-3 review).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Cancelled tool tests write an ok=False record before re-raising,Unknown-prefix rejections either record or are documented as pre-flight validation,Log semantics documented in execution_log module docstring
<!-- AC:END -->
