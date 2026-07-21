---
id: TASK-413
title: Skills editor - visible unsaved-changes feedback instead of silent veto
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 15:18'
updated_date: '2026-07-21 15:43'
labels:
  - skills
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P0-adjacent from the 2026-07-21 Skills UX/NNG review (verified live). Where the dirty-edit guard does fire (Back to list button and skill-row-to-row switches) it is a silent veto: the click does nothing - no toast, no status change, no prompt. The Back button appears broken. Combined with task-412 the surface is incoherent: some exits silently block, others silently discard. One consistent pattern is needed across all skill-editor exits. NNG heuristic 1 (visibility of system status).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Attempting to leave the skill editor with unsaved changes produces visible feedback explaining why navigation was blocked,User has an explicit path to leave without saving (discard affordance or prompt) as well as to save,The same pattern applies to Back to list and skill-row switches and (with task-412) rail-row switches,Behavior covered by tests
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: harness tests - Back veto emits a notification and editor stays; rail-switch veto notifies; Discard button disabled until dirty then leaves without saving\n2. Add _notify_skill_dirty_veto helper; call from all three veto sites (back, row switch, rail switch)\n3. Add 'Discard changes' button to editor action row (disabled until dirty, live-enabled by _mark_library_skill_dirty, re-disabled on save success); handler = Back body minus veto\n4. Run skills canvas + shell suites
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two-part fix. (1) Every dirty-veto site now reports: new _notify_skill_dirty_veto helper (toast LIBRARY_SKILL_DIRTY_VETO_COPY 'Unsaved skill changes — Save or Discard changes first.', warning severity to match trust toasts) called from handle_library_skill_back, handle_library_skill_row, the task-412 rail-switch guard, and flush_pending_work (screen tab-away, whose app-level caller only logs the veto; notes keep their conflict banner and prompts predate the pattern so only the skill veto reports there). (2) Explicit leave-without-saving path: 'Discard changes' button in the editor action row between Save and Delete, disabled until dirty; dirty-marking and the no-recompose save-success tail patch its disabled state in place via _set_library_skill_discard_enabled; handler reuses the clean-Back exit tail (reset + snapshot refresh + recompose). Canvas gained a dirty kwarg for the initial render. Tests: 4 new harness tests (Back veto notifies, rail veto notifies, flush_pending_work veto notifies, discard disabled-until-dirty then leaves without saving) all watched fail first; existing fake-based row-veto test strengthened to assert the notify. Skills canvas suite 45 passed.
<!-- SECTION:NOTES:END -->
