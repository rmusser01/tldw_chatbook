---
id: TASK-35080
title: Fix false default-workspace note and tell the model it has no project-folder
  access
status: Done
created_date: 2026-10-16 02:15
priority: high
updated_date: 2026-10-16 03:01
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User report: the Console model could not see project files or create a folder, and hallucinated when pressed. Root cause 1: default Console sessions carry workspace_id 'global', which workspace_context_note does not recognize as the default case (it only checks DEFAULT_WORKSPACE_ID 'workspace-default'), so get_workspace('global') returns None and the note degrades to the false one-liner 'This session is NOT running in the default workspace. (Workspace details are currently unavailable.)'. Root cause 2: in the default/global case the run offers zero fs_*/git_* tools and the note says nothing honest about that, so the model has neither capability nor awareness and confabulates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] workspace_context_note returns an honest default-workspace note for DEFAULT_WORKSPACE_ID, CONSOLE_GLOBAL_WORKSPACE_ID ('global') and None instead of '' or the false unavailable note
- [x] The default note states no project folders are bound and no fs_*/git_* workspace tools are available this run, keeps the scratch-relative-tools line when advertised, and directs the model to say so and suggest binding folders in a named workspace
- [x] Tests pinning the old empty-default-note contract are updated
- [x] Targeted tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend workspace_context_note: treat DEFAULT_WORKSPACE_ID, CONSOLE_GLOBAL_WORKSPACE_ID ('global') and None as the default case emitting an honest no-folders note (no fs_*/git_* this run; scratch-relative line when advertised; remediation: bind folders in a named workspace)\n2. Update tests pinning the old empty-string contract (test_workspace_file_roots.py) and the console note contract tests\n3. Run targeted tests\n\nADR required: no\nADR path: N/A\nReason: prompt-note behavior within the existing workspace-note model (TASK-33940.1 / settings-workspaces spec 3); no architectural boundary changes
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the honest default-workspace note in workspace_context_note (global/None/default all routed to _default_workspace_note, conditioned on advertised aliases); updated pinning tests and the bridge end-to-end test; docs updated.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
