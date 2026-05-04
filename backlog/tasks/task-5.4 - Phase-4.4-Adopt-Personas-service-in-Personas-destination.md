---
id: TASK-5.4
title: Phase 4.4 Adopt Personas service in Personas destination
status: Done
assignee: []
created_date: '2026-05-04 06:30'
labels:
  - unified-shell
  - phase-4
  - personas
  - service-adoption
dependencies: []
parent_task_id: TASK-5
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the top-level Personas destination use the existing character/persona scope service so users can see whether local characters and persona profiles are available and stage concrete behavior context into Console instead of generic placeholder copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Personas route lists a local behavior snapshot from character and persona profile services when available
- [x] #2 Personas route shows honest empty and service-unavailable recovery states
- [x] #3 Attach to Console stages concrete persona and character summary rather than generic placeholder copy
- [x] #4 Focused automated tests cover Personas available empty error handoff and tracking behavior
- [x] #5 QA evidence documents functional behavior visual usability residual risks and verification output
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing destination shell tests for Personas local behavior snapshot available empty error and concrete Console handoff behavior.
2. Implement the smallest `PersonasScreen` local snapshot loader using `character_persona_scope_service.list_characters()` and `character_persona_scope_service.list_persona_profiles()`.
3. Render local character/profile counts and sample names with stable selectors while preserving the existing legacy Personas route button.
4. Disable `Attach to Console` when no concrete local behavior context exists and keep service failures distinct from empty Personas state.
5. Build `ChatHandoffPayload` body from actual listed character and persona profile counts and sample names.
6. Add Phase 4.4 QA evidence and roadmap links.
7. Run focused Personas destination service tests plus git diff checks before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added local Personas snapshot loading in `PersonasScreen` through `character_persona_scope_service.list_characters()` and `character_persona_scope_service.list_persona_profiles()`.
- Rendered loading, available, empty, service-unavailable, and policy-denied recovery states while preserving the existing `Open Personas` route.
- Changed `Attach to Console` from a generic placeholder to a disabled-until-ready `personas-context` Chat handoff with concrete local character/profile counts, sample names, and descriptions.
- Added focused destination shell and Console handoff regression coverage plus Phase 4.4 QA evidence and roadmap links.
- Verified with `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_shell_product_model_visibility.py Tests/Character_Chat/test_character_persona_scope_service.py -q` resulting in `161 passed, 8 warnings in 92.93s`.
<!-- SECTION:NOTES:END -->
