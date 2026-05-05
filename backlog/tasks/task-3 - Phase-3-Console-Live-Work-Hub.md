---
id: TASK-3
title: 'Phase 3: Console Live Work Hub'
status: In Progress
assignee: []
created_date: '2026-05-03 14:47'
updated_date: '2026-05-05 00:19'
labels:
  - unified-shell
  - phase-3
  - console
  - agentic-control
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Console the primary live-agent control surface for workflows, schedules, ACP, MCP, RAG, artifacts, and active work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Console receives live work from the relevant product sources.
- [ ] #2 Launch, follow, status, and recovery flows are verified in the running app.
- [ ] #3 Console remains fast for repeated power-user use.
- [ ] #4 Durable QA summary exists under Docs/superpowers/qa/unified-shell/phase-3/.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Treat TASK-3.1 through TASK-3.10 as implemented Console live-work slices.
2. Keep parent TASK-3 open until TASK-3.11 completes maturity-gate QA in the running app.
3. Use TASK-3.11 to decide whether launch follow status and recovery workflows are verified across implemented sources or need explicit follow-up blockers.
<!-- SECTION:PLAN:END -->
