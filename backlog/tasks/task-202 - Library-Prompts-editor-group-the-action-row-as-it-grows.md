---
id: TASK-202
title: 'Library Prompts editor: group the action row as it grows'
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-12 22:21'
updated_date: '2026-08-08 14:35'
labels:
  - ux
  - library
  - prompts
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Second-pass UX review of Library Prompts (2026-07-12): the editor action row now carries six flat buttons (Save, Use in Console, Export, Copy text, Duplicate prompt, Delete). As Console-injection actions land in Phase 2 it will crowd; group into primary (Save) / content actions / lifecycle (Duplicate, Delete).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Normal editor actions retain stable IDs and keyboard order: Save, Use in Console, Export, Copy Markdown, Duplicate, Delete.
- [ ] #2 Actions are grouped by primary, content, and lifecycle purpose; Save is visually distinguishable and Delete uses the existing danger treatment.
- [ ] #3 Conflict actions Save as new and Reload render in the same always-visible action area.
- [ ] #4 At 200x50 the action area is visible and nonzero; at shorter sizes it remains scroll-reachable without obscuring the final editor field.
- [ ] #5 Copy Markdown copies the live unsaved working copy through the application clipboard seam and only reports success after copying succeeds.
- [ ] #6 Unavailable or failed clipboard support is reported honestly without logging Prompt bodies.
- [ ] #7 Single delete requires confirmation and warns that both the saved artifact and unsaved working copy are discarded when the editor is dirty.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement grouped, always-visible editor actions; wire live Copy Markdown; add shared delete confirmation; verify TCSS geometry. ADR required: no; ADR path: N/A; reason: UI-only change under ADR-011/ADR-040.
<!-- SECTION:PLAN:END -->
