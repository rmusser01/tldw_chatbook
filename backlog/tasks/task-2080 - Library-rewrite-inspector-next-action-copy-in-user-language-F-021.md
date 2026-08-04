---
id: TASK-2080
title: 'Library: rewrite inspector next-action copy in user language (F-021)'
status: In Progress
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-04 06:23'
labels:
  - ux-review
  - library
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
'Library remains a hub; Notes, Media, Search/RAG, and Study own deeper work.' is architecture-talk in user chrome. Evidence: library_screen.py:324-331. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Inspector next-action copy reads as user guidance, not architecture notes,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (dead-code deletion + pin). Finding from code inspection: LIBRARY_INSPECTOR_EMPTY_NEXT_ACTION_COPY and LIBRARY_INSPECTOR_EMPTY_COPY are DEAD -- the inspector pane was retired with the legacy workbench chrome (no widget composes #library-source-inspector; test_product_maturity_phase6_recovery_docs.py:279 documents the retirement), so the architecture-talk copy never renders. Wire-or-delete per the F-010 discipline: delete both constants; the user-facing guidance already lives in the F-013 landing copy ('Search everything, pick a section on the left, or add something new.') and the F-010 hub. Steps: 1. RED pin test (Tests/UI/test_library_shell.py, mirroring test_library_dead_hub_helpers_are_removed): both constants absent from library_screen module. 2. Delete the two constants. 3. Confirm no other references; run shell suite + parity + ruff.
<!-- SECTION:PLAN:END -->
