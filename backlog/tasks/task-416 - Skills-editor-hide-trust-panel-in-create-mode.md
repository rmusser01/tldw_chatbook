---
id: TASK-416
title: Skills editor - hide trust panel in create mode
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 15:18'
updated_date: '2026-07-21 16:10'
labels:
  - skills
  - ux
  - trust
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P1 from the 2026-07-21 Skills UX/NNG review (verified live). The create-skill editor renders the trust panel with 'Trust: trusted' and Unlock / Review changes / Approve buttons for a skill that does not exist on disk yet. Nothing has been trusted; the state line is false. After first save the panel correctly shows the real state. NNG heuristic 1 (visibility of system status).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The trust panel is not rendered (or shows an honest pre-save placeholder) while creating a skill that has never been saved,After the first successful save the real trust state renders as today,Covered by tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
LibrarySkillsListCanvas._compose_editor now skips _compose_trust_panel while is_create - a never-saved skill has no on-disk files, so the panel could only render a false 'Trust: trusted' state with dead Unlock/Review/Approve buttons (verified live). After the first save, _apply_library_skill_save_success's _refresh_local_source_snapshot lands a recompose with is_create=False (the same recompose that flips the rename hint on), rendering the real trust panel for the just-saved skill; the in-place _render_library_skill_trust_panel call in the save tail already NoMatches-passes harmlessly in the gap. Task-414's in-place patch test was reshaped to the existing-skill form (create mode legitimately has no panel to patch now). 2 new canvas tests (create renders none / existing still renders) watched fail first; skills canvas 61 passed, Tests/Skills 129 passed.
<!-- SECTION:NOTES:END -->
