---
id: TASK-31724
title: Retain the Database Notes editor while admitting Folder Files
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:45'
updated_date: '2026-09-05 18:53'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent navigation invalidation from repainting the outgoing retained Database authority during a Files source handoff, preserving both independent editors and focus ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Files admission invalidates stale Notes navigation without recomposing the outgoing Database editor
- [x] #2 Database and Files editor identities and original return-focus matrix remain intact across source handoffs
- [x] #3 Deterministic no-repaint regression and related return tests pass with no screen budget increase
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the outgoing Work-pane repaint caused by Files admission navigation invalidation before any return occurs.
2. Use the existing render=False option for that source handoff; retain generation cancellation, flush, leave guards and focus evacuation.
3. Replace the temporary stack probe with a deterministic no-repaint assertion and preserve all existing identity/focus/typing/undo assertions.
4. Run the original seven return cases plus related source-handoff tests and static checks; document remaining independent scroll issues separately.
ADR required: no
ADR path: N/A
Reason: Existing navigation invalidation option restores the already specified retained-authority contract; no new lifetime or focus policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Files source admission now calls the existing navigation invalidation with render=False. It still revokes old generation work but does not repaint the outgoing retained Database authority. The deterministic regression observes Work-pane recompose count (red1 before, green0 after) and original editor identity immediately after Files opens, in addition to unchanged return/typing/undo/focus assertions.
Exact eight-case retained-authority/return selection:8 passed152 deselected in23.18s. The extended nine-case run had8 passed and only the separately tracked wide scroll6vs7 mismatch; no missing receipt exceptions or editor retention failures remain. Screen size stays41305lines/1301methods.
Test Ruff and touched-function format pass; screen findings are unchanged pre-existing import findings. git diff --check and parent review pass. ADR required:no; existing non-rendering invalidation option implements the retained-authority contract. Incident recorded in library-decomposition-recipe section25. Broader responsive-scroll work and final full Notes verification remain open separately.
<!-- SECTION:NOTES:END -->
