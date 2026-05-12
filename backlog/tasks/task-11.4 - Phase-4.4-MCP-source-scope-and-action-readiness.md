---
id: TASK-11.4
title: 'Phase 4.4: MCP source scope and action readiness'
status: To Do
assignee: []
created_date: '2026-05-12 00:00'
labels:
  - product-maturity
  - phase-4-agent-execution
  - mcp
dependencies:
  - TASK-11.1
references:
  - Docs/superpowers/specs/2026-04-21-unified-mcp-control-plane-parity-design.md
  - Docs/superpowers/plans/2026-05-12-phase-4-agent-configuration-execution.md
parent_task_id: TASK-11
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make MCP source scope server hierarchy and action readiness clear while preserving the Unified MCP control plane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 MCP presents local/server scope and server-first hierarchy without flattening tools into generic settings.
- [ ] #2 User can tell which server or tool actions are ready blocked or policy-denied.
- [ ] #3 Existing MCP route aliases and Unified MCP panel behavior remain compatible.
- [ ] #4 QA walkthrough and focused regression evidence prove the MCP flow is usable in the running app.
<!-- AC:END -->
