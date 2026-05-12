---
id: TASK-11.3
title: 'Phase 4.3: Skills attach validation and local execution contract'
status: To Do
assignee: []
created_date: '2026-05-12 00:00'
labels:
  - product-maturity
  - phase-4-agent-execution
  - skills
dependencies:
  - TASK-11.1
references:
  - Docs/superpowers/specs/2026-05-01-local-skills-kanban-parity-design.md
  - Docs/superpowers/plans/2026-05-12-phase-4-agent-configuration-execution.md
parent_task_id: TASK-11
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make local Agent Skills understandable and attachable to Console with validation and honest execution readiness.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 User can distinguish valid local skills from invalid missing or policy-denied skill states.
- [ ] #2 Valid selected skill can stage context into Console without mutating the user skill directory.
- [ ] #3 Skill validation follows the Agent Skills `SKILL.md` metadata contract for the visible readiness state.
- [ ] #4 QA walkthrough and focused regression evidence prove the Skills flow is usable in the running app.
<!-- AC:END -->
