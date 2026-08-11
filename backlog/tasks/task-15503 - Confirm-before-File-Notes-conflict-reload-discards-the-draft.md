---
id: TASK-15503
title: Confirm before File Notes conflict reload discards the draft
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 20:56'
updated_date: '2026-08-11 21:46'
labels:
  - notes
  - filesystem
  - recovery
  - ux
  - data-safety
dependencies: []
references:
  - >-
    backlog/tasks/task-399.8.2 -
    B1b2-Build-bounded-conflict-comparison-and-resolution-UX.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
  - >-
    .impeccable/critique/2026-08-11T20-58-28Z__ok-widgets-library-library-file-notes-workspace-py.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The current conflict and error recovery surface says the draft is preserved, but activating Reload immediately replaces the editor with disk bytes. Add an explicit, keyboard-safe destructive confirmation now, while the broader three-sided comparison and resolution experience remains tracked by TASK-399.8.2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 In conflict or error states, the reload action explicitly says that it will discard the current draft and load disk bytes.
- [x] #2 The first reload activation never replaces editor contents; it opens a distinct confirmation state whose safe default is cancel.
- [x] #3 Cancel and Escape close confirmation, preserve the exact draft and conflict state, and return focus to the reload opener.
- [x] #4 Confirm revalidates the active root, file identity, session generation, and current disk state before replacing the draft; a stale or unavailable target fails closed with actionable copy.
- [x] #5 Mounted tests prove preservation on first activation and cancellation, intentional replacement only after confirmation, keyboard operation, and complete copy at 40x20 and a normal width.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterize conflict and error reload through the mounted production Library shell, including narrow geometry, opener focus, Escape routing, and stale identity seams.
2. Add an inline retained confirmation with explicit destructive copy, Cancel as the safe default, and truthful footer help.
3. Capture the current disk snapshot when confirmation opens, then revalidate root binding, file identity, session generation, save state, and disk hash before intentional replacement; fail closed with actionable copy.
4. Add mounted 40x20 and normal-width keyboard tests plus stale/unavailable-target coverage, then update File Notes documentation.
5. Run focused tests, Ruff, compileall, diff review, and complete the Backlog record.

ADR required: no

ADR path: N/A (ADR-021 and ADR-031 apply)

Reason: This adds the confirmation and revalidation behavior already required by the accepted File Notes disk-authority and keybinding decisions; no storage, ownership, or service boundary changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a retained inline confirmation before conflict or error reload can replace the File Notes draft. The destructive opener now reads **Discard draft and reload**. Its first activation captures the current disk snapshot, presents complete warning copy with **Cancel** focused, and makes the editor read-only for the decision. Cancel and Library-level Escape preserve the exact draft and save state, restore focus to the opener, and update footer/F1 help to say `esc cancel reload`.

Confirm re-reads the file and validates the active service, root binding and generation, opened-file identity, editing-session key, conflict/error state, and captured disk hash before replacement. Root, file, session, missing-target, unreadable-target, or second disk changes fail closed with recovery guidance. At 40x20 the inactive path field is temporarily hidden so the complete warning and both decisions stay within the viewport; the regular editor layout returns on cancel or completion.

The original production-shell acceptance failed on the generic **Reload** label. The first implementation then exposed the warning one row below the 40x20 viewport, and the geometry assertion drove the compact layout correction. The final focused regression matrix passed 17 tests, including production Library entry, keyboard Cancel/Escape/Confirm, identity and disk races, error-state preservation, maintenance actions, existing Files-to-Database Escape, and breakpoint round trips. A final two-size confirmation rerun also passed. Ruff passed for the workspace and tests; `library_screen.py` passed with its seven unrelated pre-existing E721 findings excluded. Compileall and `git diff --check` passed.

ADR required: no. ADR-021 and ADR-031 already define disk authority, fail-closed recovery, destructive confirmation, and truthful keyboard hints. No new storage, ownership, service, or long-lived application boundary was introduced.

Modified files: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`, `tldw_chatbook/UI/Screens/library_screen.py`, `Tests/UI/test_library_file_notes_workspace.py`, and `Docs/User_Guide/library/file-notes.md`.
<!-- SECTION:NOTES:END -->
