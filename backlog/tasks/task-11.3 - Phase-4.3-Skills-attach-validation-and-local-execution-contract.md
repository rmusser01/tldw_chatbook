---
id: TASK-11.3
title: 'Phase 4.3: Skills attach validation and local execution contract'
status: Done
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
- [x] #1 User can distinguish valid local skills from invalid missing or policy-denied skill states.
- [x] #2 Valid selected skill can stage context into Console without mutating the user skill directory.
- [x] #3 Skill validation follows the Agent Skills `SKILL.md` metadata contract for the visible readiness state.
- [x] #4 QA walkthrough and focused regression evidence prove the Skills flow is usable in the running app.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused regressions for Agent Skills metadata validation and selected-skill Console handoff.
2. Extend the local Skills service to parse and report Agent Skills frontmatter validation without rejecting or mutating invalid skills.
3. Render valid and invalid skill readiness in the Skills destination and stage only the selected valid skill.
4. Capture an actual textual-web screenshot for user approval and run focused Skills regressions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added Agent Skills metadata validation to the local Skills service and surfaced validation status/errors in the Skills destination. The screen now defaults to the first valid local skill, lets users choose a listed skill, disables Console attach for invalid selections, and stages only the selected valid skill with stable handoff metadata. QA evidence is recorded in `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-3-skills-attach-validation.md` with a user-approved actual textual-web screenshot.
<!-- SECTION:NOTES:END -->
