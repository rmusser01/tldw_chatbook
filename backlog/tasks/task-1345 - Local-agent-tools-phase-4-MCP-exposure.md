---
id: TASK-1345
title: 'Local agent tools phase 4: MCP exposure'
status: In Progress
assignee: []
created_date: '2026-08-05 23:38'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md §3.1. Plan: Docs/superpowers/plans/2026-08-05-local-agent-tools-phase4.md. ADRs 032/033. Route through LocalToolProvider gate; todo_write not exposed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 External MCP clients can call allowed local tools through the server (invocation routed through LocalToolProvider's gate)
- [ ] #2 ask-state tools fail closed externally with an external-appropriate refusal (no approval card exists outside the Console)
- [ ] #3 Kill switch and deny states honored; operator grants (always-allow from Console) enable external use
- [ ] #4 Exposure gated behind [mcp] expose_local_tools (default false); todo_write not exposed (documented)
- [ ] #5 All new tests pass
<!-- AC:END -->


## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-05-local-agent-tools-phase4.md
<!-- SECTION:PLAN:END -->
