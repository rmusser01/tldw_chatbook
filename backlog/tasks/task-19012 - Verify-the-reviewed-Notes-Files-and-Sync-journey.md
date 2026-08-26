---
id: TASK-19012
title: Verify the reviewed Notes Files and Sync journey
status: Done
assignee:
  - '@codex'
created_date: '2026-08-20 07:53'
updated_date: '2026-08-21 14:41'
labels:
  - notes
  - ux
  - accessibility
  - verification
dependencies:
  - TASK-19001
  - TASK-19002
  - TASK-19011
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close compact-layout, accessibility, documentation, and live isolated evidence across Library notes, Folder files, Import once, lasting sync recovery, and Session Git without expanding storage or Git policy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Production-shaped mounted tests cover Library notes, Folder files, Import once, lasting setup/review/attention/recovery, and Session Git at wide, 60x20, and 40x20 where supported.
- [x] #2 Every focus target, disclosure, cancellation path, footer hint, text-explicit state, disabled action, error, and recovery action is reachable and meets measured contrast requirements.
- [x] #3 Storage authority, running work, latest durable outcome, and next actions remain truthful through navigation, compact transitions, restart, cancellation, and partial failure.
- [x] #4 Isolated live TUI verification uses a scratch config and data directory, captures rendered frames, and does not open or migrate the user’s real databases.
- [x] #5 User guides, feature docs, screenshots or text captures, Backlog tasks, and ADR-059/073 links match shipped behavior; server-backed lasting sync remains documented as unavailable.
- [x] #6 Focused and broader automated tests, CSS generation/parity, static checks, and diff checks pass or any inherited baseline failure is reproduced on the untouched base with exact evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a production-shaped mounted journey matrix using the real `LibraryScreen` hierarchy and `TldwCli.CSS_PATH`, with wide, 60x20, and supported 40x20 render/focus/accessibility assertions.
2. Exercise Notes, Folder Files, Import once, lasting setup/review/attention/recovery, and Session Git through their physical messages and outermost admitted paths, including restart, cancellation, partial-failure, and truthful next-action states.
   Retarget the retained Import-once integration oracles from the removed legacy toolbar selectors to the shipped Add-from-files chooser; do not restore a legacy compatibility surface.
3. Build a safe live-TUI verifier that creates a disposable HOME/config/data/profile, seeds only scratch fixtures, captures rendered frames and checksums, and tears down without touching the caller's databases.
4. Inspect live captures top-to-bottom, update user/feature documentation and any real verification lesson, then run CSS parity, focused and broad automated gates, compilation, static checks, and diff checks.
5. Record exact automated/live evidence, audit TASK-19000 through TASK-19012 status, obtain independent review, and close/commit only if every acceptance criterion is satisfied.

ADR required: no new ADR
ADR path: backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md; backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md; backlog/decisions/035-file-notes-session-git-index-controls.md; backlog/decisions/038-file-notes-guarded-session-commit.md; backlog/decisions/039-file-notes-guarded-session-push.md; backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md; backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md
Reason: TASK-19012 verifies and documents already-decided behavior without changing storage, sync, Git, or ownership policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a production-shaped journey matrix over the real `TldwCli`/`LibraryScreen` hierarchy. It exercises Database Notes and Folder files at wide, 60x20, and supported 40x20 layouts; Import once check/cancel/receipt; lasting-sync review, activation, attention, partial recovery, and reopened private-store recovery; and physical Session Git stage/commit/push cancellation and result paths.
- Retargeted retained import-flow tests and UI copy from the removed legacy toolbar to the shipped Add from files chooser. Fixed retained-import focus restoration and limited runtime-ready canvas refresh to the mounted Database Notes list so unrelated warm routes do not recompose.
- Added `Helper_Scripts/verify_notes_files_sync_tui.py`. The helper uses a scrubbed allowlisted environment, isolated HOME/XDG/config/data directories, bounded subprocesses, offline model downloads, byte-stability sentinels for generated repository assets, unchanged decoy checksums, rendered-frame checksums, and scratch teardown. The final live run passed with five inspected frames in `/private/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-notes-files-sync-evidence-edbpc584`; the decoy hash remained `ec0c8695...b71a2` and all five repository sentinels were byte-identical.
- Updated the Notes user/feature guides and recorded the verification incident in `backlog/docs/lessons-live-verification.md`. ADR-021/031/035/038/039/059/073 remain the governing decisions; no new ADR was required.
- Verification: the exact broad programme gate passed `2112 passed, 16 warnings in 2704.31s`; CSS source/bundle parity passed; `compileall` passed with one pre-existing invalid-escape warning; Ruff passed on all owned production/test/helper paths; both new files are Ruff-formatted; and `git diff --check` passed. Independent review reported no Critical, Important, or blocking Minor findings and marked the implementation Ready.
- Audited TASK-19000 through TASK-19011 as Done before closing TASK-19012. No inherited test failure required a base comparison because the prescribed gate was fully green.
<!-- SECTION:NOTES:END -->

## Decision Record Check

ADR required: no new ADR
ADR paths: `backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md`, `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`, `backlog/decisions/035-file-notes-session-git-index-controls.md`, `backlog/decisions/038-file-notes-guarded-session-commit.md`, `backlog/decisions/039-file-notes-guarded-session-push.md`, `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`
Reason: this closes verification and documentation for already-decided behavior without adding architecture.
