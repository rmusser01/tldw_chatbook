---
id: TASK-15601
title: Add bounded File Notes conflict resolution choices
status: Done
assignee:
  - '@codex'
created_date: '2026-08-12 02:28'
updated_date: '2026-08-12 02:58'
labels:
  - notes
  - library
  - ux
  - recovery
dependencies:
  - TASK-15532
references:
  - >-
    .impeccable/critique/2026-08-11T20-58-28Z__ok-widgets-library-library-file-notes-workspace-py.md
documentation:
  - backlog/decisions/029-file-notes-disk-authority.md
  - Docs/User_Guide/library/file-notes.md
priority: high
type: enhancement
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Help users leave a File Notes conflict intentionally through explicit safe choices without implying that comparison itself resolves anything or exposing an overwrite path that lacks durable recovery guarantees.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Conflict state exposes one explicit resolution disclosure while routine states do not
- [x] #2 The resolution surface names Keep editing Save draft as new note and Discard draft and load disk without exposing overwrite
- [x] #3 Keep editing closes the surface returns focus safely and preserves Base Draft Disk and conflict state
- [x] #4 Save draft as new note keeps the existing no-clobber path validation and exact body preservation then opens the created note only after success
- [x] #5 Discard draft and load disk retains the distinct Cancel-first confirmation and all existing freshness revalidation
- [x] #6 The resolution surface remains readable keyboard reachable and focus safe at 40x20 and 120x40
- [x] #7 Focused mounted tests static checks documentation and self-review pass without changing disk authority save publication or recovery protocols
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; conform to `backlog/decisions/029-file-notes-disk-authority.md`.
Reason: this reorganizes existing safe-copy and confirmed-reload behaviors behind an explicit conflict-resolution disclosure and adds a no-op Keep editing choice; it adds no overwrite policy, storage contract, or mutation primitive.

1. Add failing mounted regressions for conflict-only disclosure, complete choice labels, safe focus, and compact geometry.
2. Route Keep editing to a non-resolving close path, Save draft as new note to the existing no-clobber copy flow, and Discard draft and load disk to the existing revalidated confirmation.
3. Preserve comparison as a peer decision-support action and keep overwrite absent from code, copy, and documentation.
4. Run focused and complete affected File Notes tests plus static, diff, and backlog checks.
5. Record evidence, complete the task, and prepare the atomic PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a conflict-only **Resolve conflict** disclosure with **Keep editing**, **Save draft as new note**, and **Discard draft and load disk**. **Compare** remains available and no overwrite action is exposed.
- Reused the existing exact no-clobber copy operation for new-note recovery and the existing Cancel-first, identity- and freshness-revalidated reload confirmation for discard. Cancel now returns focus to whichever destructive action opened that confirmation.
- Added mounted coverage for both 40x20 and 120x40 layouts, safe focus, exact body-style preservation, occupied-destination refusal, and the existing reload-confirmation path. Updated the File Notes user guide.
- ADR required: no. The implementation conforms to `backlog/decisions/029-file-notes-disk-authority.md` and changes no disk authority, storage, autosave publication, or recovery protocol.
- Verification: `pytest Tests/UI/test_library_file_notes_workspace.py` (86 passed); adjacent conflict/disclosure set (24 passed); Ruff on both touched Python files; `compileall`; `git diff --check`. One suite-order focus timing assertion failed once, then passed in isolation and in the clean full-module rerun.
- Self-review found no additional actionable issue. No generalizable new testing or architecture lesson was needed.
<!-- SECTION:NOTES:END -->
