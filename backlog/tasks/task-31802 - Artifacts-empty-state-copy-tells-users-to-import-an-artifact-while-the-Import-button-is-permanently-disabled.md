---
id: TASK-31802
title: >-
  Artifacts empty-state copy tells users to 'import an artifact' while the
  Import button is permanently disabled
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-05 19:15'
updated_date: '2026-09-06 14:50'
labels:
  - bug
  - ux
  - artifacts
  - copy
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). The Artifacts empty state instructs importing an artifact, but the only Import button is disabled with no explanation of how to enable it. Either enable the path, explain the precondition, or change the copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Empty-state guidance matches an action the user can actually take.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: Import button (#artifacts-import-artifact) is disabled=True permanently while the empty-state detail copy says 'import an artifact'.\n2. Fix copy: drop 'import an artifact' from #artifacts-detail-empty; add a visible inline precondition note (#artifacts-import-note) under the disabled button.\n3. RED test asserting the note paints and the empty copy no longer says 'import an artifact'.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced: #artifacts-import-artifact is permanently disabled while #artifacts-detail-empty told users to 'import an artifact'.

Fix (TASK-31802): removed 'import an artifact' from the empty-state detail copy (now 'Create a Chatbook in Console, or use Library sources to generate outputs') and added a visible inline note #artifacts-import-note under the disabled button stating 'Import Artifact is not yet available in this shell.' Both acceptable remedies applied; the import path itself remains a later-stage feature.

Test: test_import_precondition_is_explained_and_empty_copy_is_honest (asserts the button stays disabled, the note paints, and the empty copy no longer says 'import an artifact').

Files: tldw_chatbook/UI/Screens/artifacts_screen.py, Tests/UI/test_artifacts_screen_reports.py, Docs/User_Guide/artifacts.md.
<!-- SECTION:NOTES:END -->
