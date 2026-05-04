---
id: TASK-5.2
title: Phase 4.2 Adopt Skills services in Skills destination
status: Done
assignee: []
created_date: '2026-05-04 04:08'
updated_date: '2026-05-04 04:17'
labels:
  - unified-shell
  - phase-4
  - skills
  - service-adoption
dependencies: []
parent_task_id: TASK-5
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the top-level Skills destination use the existing skills scope service so users can see local Agent Skills, recover from empty or failed service states, and stage concrete skill context into Console instead of generic placeholder copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Skills route lists local skills from skills_scope_service when available
- [x] #2 Skills route shows honest empty and service-unavailable recovery states
- [x] #3 Attach to Console stages selected or listed skill context rather than generic placeholder copy
- [x] #4 Focused automated tests cover Skills available empty error and tracking behavior
- [x] #5 QA evidence documents functional behavior visual usability residual risks and verification output
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing destination shell tests for Skills service-backed available empty error and concrete Console handoff behavior.
2. Implement the smallest SkillsScreen service loader using skills_scope_service.list_skills(mode='local') with safe empty and error states.
3. Render local skill summaries with stable selectors and disable Attach to Console when there is no concrete skill context.
4. Build ChatHandoffPayload body from actual listed skill names descriptions and metadata when context exists.
5. Add Phase 4 QA evidence and roadmap links.
6. Run focused Skills destination service tests plus git diff checks before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented local Skills destination service adoption by loading local Agent Skills through `skills_scope_service.list_skills(mode="local")`, rendering available, empty, loading, and service-error states, and disabling Console handoff until concrete local skill context exists. The Console handoff now stages a `skills-context` payload built from listed skill names, descriptions, argument hints, record IDs, backend metadata, and the local skills directory instead of the old generic placeholder body.

Updated destination and Console handoff tests to cover available, empty, error, concrete handoff, and tracking evidence behavior. Added Phase 4.2 QA evidence and roadmap/README links. PR review follow-up added coverage for sanitized skill text, missing-service wiring, policy-denied recovery copy, typed initializer annotations, and the named Skills page-size constant. Focused verification: `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_console_live_work_handoffs.py Tests/Skills/test_skills_scope_service.py Tests/Skills/test_local_skills_service.py -q` -> `110 passed, 8 warnings in 69.45s`.
<!-- SECTION:NOTES:END -->
