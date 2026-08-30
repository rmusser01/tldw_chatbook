---
id: TASK-24601
title: Live work sources reports hardcoded Connected for five of seven rows
status: To Do
assignee: []
created_date: '2026-08-30 00:53'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Under a heading that reads as measured readiness, Watchlists, Workflows, Schedules, RAG and Artifacts are literal status="Connected" string constants; only ACP derives from a runtime snapshot and MCP is a constant "Not wired". from_acp_runtime_status is the sole builder, so no code path ever measures the other five. A user who discovers this has reason to distrust every other status line in the rail.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No row in Live work sources displays a readiness word that was not derived from a runtime check
- [ ] #2 Sources that are not probed render an explicit not-checked state rather than Connected
- [ ] #3 A test fails if a readiness row's status is a literal in the builder rather than derived from an input
<!-- AC:END -->
