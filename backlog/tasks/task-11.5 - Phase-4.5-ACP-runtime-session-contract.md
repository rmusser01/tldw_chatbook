---
id: TASK-11.5
title: 'Phase 4.5: ACP runtime session contract'
status: To Do
assignee: []
created_date: '2026-05-12 00:00'
labels:
  - product-maturity
  - phase-4-agent-execution
  - acp
dependencies:
  - TASK-11.1
references:
  - Docs/superpowers/plans/2026-05-12-phase-4-agent-configuration-execution.md
parent_task_id: TASK-11
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make ACP runtime setup session readiness and Console follow states explicit without moving ACP ownership into Settings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ACP shows whether an ACP-compatible runtime is configured and what setup step is needed next.
- [ ] #2 Session and Console-follow actions are enabled only when a real session payload is available.
- [ ] #3 Runtime ownership remains under ACP while Settings remains limited to global defaults.
- [ ] #4 QA walkthrough and focused regression evidence prove the ACP flow is usable or honestly blocked in the running app.
<!-- AC:END -->
