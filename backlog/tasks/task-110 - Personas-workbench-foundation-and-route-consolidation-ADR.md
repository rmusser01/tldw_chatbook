---
id: TASK-110
title: Personas workbench foundation and route consolidation ADR
status: Done
assignee: []
created_date: '2026-06-10 06:12'
updated_date: '2026-06-10 06:21'
labels:
  - personas
  - ux
  - foundation
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-20-ccp-destination-native-route-replacement.md
  - backlog/decisions/007-personas-workbench-route-consolidation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish the durable Personas workbench foundation before moving more visible UI out of the legacy CCP surface. The goal is to make the top-level Personas route the product-owned destination for characters, persona profiles, prompts, dictionaries, lore, import/export, and Console handoff while keeping the current CCP route as compatibility plumbing until later UI slices retire it safely.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ADR records the top-level Personas destination, CCP compatibility boundary, and non-goals before code changes.
- [x] #2 A reusable Personas workbench state model covers mode, selected entity, search/filter text, runtime source, loading state, dirty state, and reset behavior.
- [x] #3 A reusable Personas message vocabulary covers mode changes, entity selection, search/filter changes, create/import/export, attach/start-chat, save, cancel, and refresh events without depending on CCPScreen.
- [x] #4 Focused unit tests verify state defaults, mode switching, selection metadata, reset behavior, and message payloads.
- [x] #5 This foundation slice makes no user-visible Personas layout changes and requires no screenshot approval until a later UI slice.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/007-personas-workbench-route-consolidation.md
Reason: This task defines long-lived destination ownership, route compatibility, and UI module boundaries.

1. Create the ADR and link it from this task.
2. Add failing unit tests for reusable Personas workbench state and messages.
3. Implement the smallest state/message modules under `tldw_chatbook/Widgets/Persona_Widgets`.
4. Run focused tests and `git diff --check`.
5. Add implementation notes before marking ready for review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added ADR-007 to record Personas as the durable top-level destination and CCP as a compatibility route while reusable handlers/widgets are adapted.
- Added `tldw_chatbook/Widgets/Persona_Widgets` with screen-independent state and Textual message contracts for later destination-native panes.
- Added focused regression coverage for state defaults, mode switching, selection metadata, runtime-source reset, and message payloads.
- No visible UI layout changed in this slice, so no screenshot approval was required.
- Verification: `python -m pytest -q Tests/UI/test_personas_workbench_foundation.py --tb=short` passed; `python -m pytest -q Tests/UI/test_ccp_screen.py --tb=short` passed; `git diff --check` passed.
<!-- SECTION:NOTES:END -->
