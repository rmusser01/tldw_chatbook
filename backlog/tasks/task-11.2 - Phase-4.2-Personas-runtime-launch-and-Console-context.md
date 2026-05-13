---
id: TASK-11.2
title: 'Phase 4.2: Personas runtime launch and Console context'
status: Done
assignee: []
created_date: '2026-05-12 00:00'
labels:
  - product-maturity
  - phase-4-agent-execution
  - personas
dependencies:
  - TASK-11.1
references:
  - Docs/superpowers/specs/2026-04-19-characters-persona-runtime-alignment-vertical-design.md
  - Docs/superpowers/plans/2026-05-12-phase-4-agent-configuration-execution.md
parent_task_id: TASK-11
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Personas destination a usable local character/persona selection and Console-context surface with honest runtime readiness.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 User can identify local character and persona options and understand whether each can be attached to Console.
- [x] #2 Selected character or persona produces a stable Console handoff payload without creating a parallel persona registry.
- [x] #3 Disabled or policy-denied states explain what is blocked and what the user can do next.
- [x] #4 QA walkthrough and focused regression evidence prove the Personas flow is usable in the running app.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add mounted regression coverage for selected local character/persona targets, Console handoff metadata, and policy-denied readiness.
2. Reuse the existing character/persona scope service snapshot data to derive selected Console target state.
3. Render selected target/readiness in the Personas inspector and keep blocked states honest.
4. Capture actual textual-web screenshots and run focused Personas/service regressions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a selected behavior target model to the Personas screen using existing local character/persona snapshot records. The inspector now shows the selected runtime target, policy-denied snapshots report Console as blocked with a recovery reason, and Console handoff payloads include stable selected target metadata without introducing a parallel registry. Focused mounted tests cover default character selection, switching to a persona profile, blocked policy state, and the existing destination boundary assertion was made case-insensitive after full-file verification exposed a casing-only failure. Actual textual-web screenshots were captured for both clean empty state and fixture-backed selected target state.
<!-- SECTION:NOTES:END -->
