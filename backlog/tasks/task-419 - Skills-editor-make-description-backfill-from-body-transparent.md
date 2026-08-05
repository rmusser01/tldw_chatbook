---
id: TASK-419
title: Skills editor - make description backfill from body transparent
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 15:19'
updated_date: '2026-07-21 16:42'
labels:
  - skills
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2 from the 2026-07-21 Skills UX/NNG review (verified live). Saving a new skill with an empty Description silently copies body-derived text into the Description field; the user typed only a body and finds the description populated after save with no explanation. Silent field mutation. NNG heuristics 1 and 3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When a save would derive the description from the body the user can tell (field left empty with the derived value clearly marked as automatic or an inline note explaining the backfill),No silent mutation of user-visible fields on save,Covered by tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: LocalSkillsService backfills a missing frontmatter description from the skill's first body line for LIST display; build_skill_editor_state then echoed that derived text into the Description Input as if the user had written it - and the next save would have ratcheted it into the on-disk frontmatter. Fix at the state layer with an exact discriminator (no heuristics): build_skill_editor_state now parses the frontmatter it already splits and sets new SkillEditorState.description_derived=True (Description field kept EMPTY, matching the file) iff the record carries a description the frontmatter lacks. The canvas renders an explanatory hint under the empty field ('No description set - lists show the skill's first body line automatically. Type here to set your own.', #library-skill-description-hint). Saves now write exactly what the field shows - no silent mutation and no ratchet. 4 new tests (2 state-layer, 2 canvas render) watched fail first; canvas+state 83 passed, Tests/Skills 129 passed.
<!-- SECTION:NOTES:END -->
