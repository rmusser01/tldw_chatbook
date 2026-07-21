---
id: TASK-424
title: Skills editor - keyboard accelerators and form polish
status: To Do
assignee: []
created_date: '2026-07-21 15:19'
labels:
  - skills
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2 from the 2026-07-21 Skills UX/NNG review. Everything on the Skills surface is mouse-button-only: Save/Delete/trust actions require scrolling past the whole form; no save/back accelerators; the create editor does not focus the Name field; name format (lowercase/numbers/hyphens) is only validated at save although the rule is known upfront; list rows separate the name button from its flags/description line by two blank rows weakening association. NNG heuristics 7 (flexibility and efficiency) and 5 (error prevention).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Save is reachable via a keyboard binding from anywhere in the editor,Back/escape leaves the editor honoring the unsaved-changes guard,Create editor focuses the Name field on open,Name format guidance is visible before save (placeholder or hint) and invalid names still error as today,List rows render name and metadata as one visually associated block,Covered by tests
<!-- AC:END -->
