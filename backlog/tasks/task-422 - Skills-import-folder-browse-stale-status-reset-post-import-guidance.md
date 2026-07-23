---
id: TASK-422
title: 'Skills import - folder browse, stale status reset, post-import guidance'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 15:19'
updated_date: '2026-07-21 19:47'
labels:
  - skills
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2 from the 2026-07-21 Skills UX/NNG review (stale status verified live). Import row friction: (1) Browse opens a file-only picker while the placeholder invites a skill FOLDER path - the common one-directory-per-skill shape must be typed by hand; (2) the open import row and its last status/error persist across editor round-trips and resurface stale minutes later; (3) success copy says 're-review it in the trust panel' with no way to get there from the message. NNG heuristics 7 and 1.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Importing a skill directory is possible without typing the path by hand,Returning to the list view resets or closes the import row so stale errors do not resurface,Successful import gives a direct path to the imported skill's trust panel (open row or equivalent),Covered by tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three-part polish. (1) Folder browse: new 'Browse folder…' button pushes the vendored SelectDirectory dialog (already used elsewhere in library_screen) so the common one-directory-per-skill shape no longer has to be typed by hand; file Browse… unchanged. (2) Stale-state reset: new _reset_library_skills_import_state (open/path/status/review-name) called from _reset_library_skill_editor_state (all editor exits funnel through it) and from _select_library_rail_row's fresh-entry reset block - the live-verified stale 'Please enter a file or folder path.' resurfacing across editor round-trips is gone. (3) Post-import guidance: success stores _library_skills_import_review_name and the import row renders a 'Review "<name>"…' button (#library-skills-import-review) that opens the imported skill's editor via the exact row-press flow (trust panel included); cleared on cancel/run-start/reset. 5 new tests (folder-browse render, reset clears import state, review-button render, review handler opens editor, folder handler pushes SelectDirectory) watched fail first. Canvas 79 passed, Tests/Skills 129 passed.
<!-- SECTION:NOTES:END -->
