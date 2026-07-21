---
id: TASK-420
title: >-
  Skills - remove dead chrome (inert Model override field, mis-scoped u
  shortcut)
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 15:19'
updated_date: '2026-07-21 16:45'
labels:
  - skills
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2 from the 2026-07-21 Skills UX/NNG review. Two no-op affordances on the Skills surface: (1) the Model override input is live, editable and dirty-marking but annotated 'Not applied in v1.' - it does nothing and can even trigger unsaved-changes friction; (2) the Library footer advertises 'u - use Library context in Console' on the Skills canvas but the action early-returns unless the Search/RAG row is selected. NNG heuristic 8 (aesthetic and minimalist design).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Model override is hidden or read-only-disabled until it has effect and no longer marks the editor dirty,The footer u shortcut hint appears only on rows where the action works,Covered by tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Model override: Input renders disabled=True (kept visible so an imported skill's SKILL.md model value round-trips instead of vanishing), hint copy extended to 'Not applied in v1 — shown for SKILL.md round-tripping only.', and #library-skill-model removed from the dirty-marking Input.Changed handler set (belt-and-braces - a disabled Input can't be edited). Footer u hint: _register_footer_shortcuts now registers LIBRARY_SHORTCUTS only while _library_selected_row_id == LIBRARY_ROW_BROWSE_SEARCH (the exact gate action_library_rag_use_in_console enforces) and _select_library_rail_row re-registers on every row switch - the documented personas-style dynamic-context use of the persisting footer API; empty tuple clears the hint elsewhere. Tests written after a first implementation pass (process slip), then proven RED by reverting the source patch and watching both fail before restoring: model-disabled canvas test and a harness test asserting registration is () on the Skills row and ('u', ...) on the Search row. Canvas suite 72 passed, Tests/Skills 129 passed.
<!-- SECTION:NOTES:END -->
