---
id: TASK-5.3
title: Phase 4.3 Adopt Library source services in Library destination
status: Done
assignee: []
created_date: '2026-05-04 05:36'
labels:
  - unified-shell
  - phase-4
  - library
  - service-adoption
dependencies: []
parent_task_id: TASK-5
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the top-level Library destination use existing local source services so users can see whether notes media and conversations are available and stage concrete Library source context into Console instead of generic placeholder copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library route lists a local source snapshot from notes media and conversation services when available
- [x] #2 Library route shows honest empty and service-unavailable recovery states
- [x] #3 Use in Console stages concrete Library source summary rather than generic placeholder copy
- [x] #4 Focused automated tests cover Library available empty error handoff and tracking behavior
- [x] #5 QA evidence documents functional behavior visual usability residual risks and verification output
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing destination shell tests for Library local source snapshot available empty error and concrete Console handoff behavior.
2. Implement the smallest LibraryScreen local snapshot loader using `notes_scope_service`, `media_reading_scope_service`, and `chat_conversation_scope_service`.
3. Render local source counts and sample titles with stable selectors while preserving existing legacy navigation buttons.
4. Disable Use in Console when no concrete local source context exists and keep service failures distinct from empty Library state.
5. Build ChatHandoffPayload body from actual listed source categories counts and sample titles.
6. Add Phase 4 QA evidence and roadmap links.
7. Run focused Library destination service tests plus git diff checks before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added local Library snapshot loading in `LibraryScreen` through `notes_scope_service`, `media_reading_scope_service`, and `chat_conversation_scope_service`.
- Rendered loading, available, empty, service-unavailable, and policy-denied recovery states while preserving existing legacy Library navigation routes.
- Changed `Use in Console` from a generic placeholder to a disabled-until-ready `library-source-snapshot` Chat handoff with concrete source counts and sample titles.
- Added focused destination shell and Console handoff regression coverage plus Phase 4.3 QA evidence and roadmap links.
- Verified with `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_shell_product_model_visibility.py -q` resulting in `103 passed, 1 warning in 32.21s`.
<!-- SECTION:NOTES:END -->
