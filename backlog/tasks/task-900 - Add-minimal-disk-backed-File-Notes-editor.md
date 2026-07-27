---
id: TASK-900
title: Add minimal disk-backed File Notes editor
status: Done
assignee:
  - '@codex'
created_date: '2026-07-27 14:32'
updated_date: '2026-07-27 17:29'
labels:
  - notes
  - library
  - files
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-07-27-minimal-file-notes-design.md
  - Docs/superpowers/plans/2026-07-27-minimal-file-notes.md
  - backlog/decisions/029-file-notes-disk-authority.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users manage one existing Git-backed Markdown/text folder from Library while disk and filename/path remain authoritative. Chatbook writes ordinary filesystem changes and keeps one separate SQLite replica for search and recovery without changing Database Notes or Git workflows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 User can choose and persist one notes root, browse its folder tree, and see a missing root as offline without filesystem or replica mutation.
- [x] #2 Search replaces the tree and opens matching Markdown/text files.
- [x] #3 Editor changes write directly to disk with debounced hash-checked saves, while unresolved dirty/conflict/error states prevent navigation from discarding the draft.
- [x] #4 Valid leading frontmatter, BOM, uniform newline style, and final-newline state are preserved while the body is edited; unsafe text files remain path-visible and read-only.
- [x] #5 Create, rename, move, delete, and restore operate on actual files beneath the root.
- [x] #6 Delete commits a SQLite recovery snapshot and tombstone before unlinking.
- [x] #7 SQLite is a separate root-namespaced replica with opt-in history for protected paths, and protected writes cannot occur before their pre-edit checkpoint commits.
- [x] #8 Changed this session lists only Chatbook mutations.
- [x] #9 Existing Database Notes and Git behavior remain unchanged.
- [x] #10 External changes detected before publication are never silently overwritten, reconciliation does not remount an open editor, and scanning does not block the UI.
- [x] #11 Restorable deletions remain discoverable after restart.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/029-file-notes-disk-authority.md
Reason: Implements the accepted disk-authority, independent-replica, conflict, and Library ownership boundaries.

1. Build the single SQLite replica with FTS, protected checkpoints, and tombstones.
2. Build the exact-byte, hash-checked filesystem service.
3. Mount one retained File Notes workspace in Library and delegate leave guards.
4. Run only focused File Notes tests and close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the accepted disk-authoritative File Notes design from ADR-029.

- Added a root-namespaced SQLite current replica with FTS, protected editing-session checkpoints, recovery snapshots, and persistent tombstones.
- Added path-safe exact-byte filesystem operations for scanning, opening, hash-checked atomic saving, no-clobber create/move, delete, restore, polling reconciliation, and Chatbook-only session changes.
- Added a retained Library Database | Files workspace with folder tree/search replacement, body-only editing, autosave/conflict actions, protection and recovery controls, narrow navigation, and leave guards.
- Hardened async transitions, remount autosave reconciliation, stale-result rejection, and owned replica shutdown without adding Git controls, watchers, or new dependencies.
- Focused verification: 42 tests passed; targeted Ruff passed. Full-suite/CI was intentionally outside the approved plan.
<!-- SECTION:NOTES:END -->
