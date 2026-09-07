---
id: TASK-15532
title: Add bounded Base Draft and Disk conflict comparison
status: Done
assignee:
  - '@codex'
created_date: '2026-08-12 01:31'
updated_date: '2026-08-12 01:48'
labels:
  - notes
  - library
  - ux
  - recovery
dependencies: []
references:
  - .impeccable/critique/2026-08-11T20-58-28Z__ok-widgets-library-library-file-notes-workspace-py.md
documentation:
  - backlog/decisions/029-file-notes-disk-authority.md
  - Docs/User_Guide/library/file-notes.md
priority: high
type: enhancement
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let File Notes users inspect the exact editor baseline, retained draft, and latest readable disk version before choosing a recovery action, without resolving or discarding the conflict.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Conflict state exposes Compare while routine states do not
- [x] #2 Compare captures the current Base Draft and Disk identities without changing the editor or resolving the conflict
- [x] #3 The comparison computes bounded diff output off the UI thread and labels Base to Draft and Base to Disk changes
- [x] #4 Missing or unreadable disk state is represented explicitly and oversized comparison output reports elision with exact side hashes and sizes
- [x] #5 Escape and Close return focus to Compare while preserving the draft conflict and existing destructive reload confirmation
- [x] #6 Focused mounted tests static checks and documentation pass without changing disk authority save or recovery protocols
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; conform to `backlog/decisions/029-file-notes-disk-authority.md`.
Reason: this adds bounded, read-only decision support around the existing hash-checked conflict state. It does not change disk authority, mutation publication, recovery ownership, or resolution policy.

1. Capture an immutable Base, Draft, and latest Disk snapshot under the current root, file, and editor-session identities.
2. Build bounded Base-to-Draft and Base-to-Disk unified comparisons off the UI thread, with explicit absent/unreadable and elided-content metadata.
3. Add a keyboard-readable comparison modal and conflict-only Compare action with deterministic close-to-opener focus.
4. Add focused unit and mounted regressions for exact side identity, stale-session refusal, missing disk, bounds, focus return, and non-resolution.
5. Update the File Notes guide, run focused/static/diff checks, self-review, and close the task only after every criterion is evidenced.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a conflict-only Compare action and a keyboard-readable modal that names Base, Draft, and Disk before users choose recovery. Base is the retained editor baseline, Draft is the exact current editor body, and Disk is freshly read or represented as absent/unreadable. The modal shows bounded Base-to-Draft and Base-to-Disk unified comparisons plus exact UTF-8 sizes and SHA-256 identities, and reports input/output elision explicitly.

Comparison computation runs off the UI thread. Exact root, binding, file-object, editor-session, conflict-state, and draft identities are checked before and after computation, so stale results fail closed without replacing the draft. Escape and Close dismiss the modal, return focus to Compare, and leave the existing conflict and destructive-reload confirmation unchanged.

Updated the File Notes guide and added pure/mounted coverage for labels, hashes, diff bounds, deleted Disk, stale-session refusal, 40×20/120×40 geometry, focus return, non-resolution, and worker-thread execution. Evidence: focused comparison matrix 7 passed; adjacent conflict/action/focus matrix 21 passed; complete affected File Notes module plus pure comparison tests 85 passed. Targeted Ruff, new-file format check, Python compilation, `git diff --check`, duplicate-task sweep, and self-review passed. Existing warnings were limited to the documented pydub deprecation, Windows SQLite privacy posture, and pytest-asyncio loop-scope notice.

ADR required: no. This conforms to ADR-029 and changes no disk authority, save publication, replica ownership, recovery protocol, or resolution policy. No new lessons entry was warranted.

Modified: `tldw_chatbook/Notes/file_notes_conflict_compare.py`, `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`, `Tests/Notes/test_file_notes_conflict_compare.py`, `Tests/UI/test_library_file_notes_workspace.py`, and `Docs/User_Guide/library/file-notes.md`.
<!-- SECTION:NOTES:END -->
