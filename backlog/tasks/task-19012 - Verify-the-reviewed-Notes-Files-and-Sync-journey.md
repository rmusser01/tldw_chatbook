---
id: TASK-19012
title: Verify the reviewed Notes Files and Sync journey
status: To Do
assignee: []
created_date: '2026-08-20 07:53'
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
- [ ] #1 Production-shaped mounted tests cover Library notes, Folder files, Import once, lasting setup/review/attention/recovery, and Session Git at wide, 60x20, and 40x20 where supported.
- [ ] #2 Every focus target, disclosure, cancellation path, footer hint, text-explicit state, disabled action, error, and recovery action is reachable and meets measured contrast requirements.
- [ ] #3 Storage authority, running work, latest durable outcome, and next actions remain truthful through navigation, compact transitions, restart, cancellation, and partial failure.
- [ ] #4 Isolated live TUI verification uses a scratch config and data directory, captures rendered frames, and does not open or migrate the user’s real databases.
- [ ] #5 User guides, feature docs, screenshots or text captures, Backlog tasks, and ADR-059/073 links match shipped behavior; server-backed lasting sync remains documented as unavailable.
- [ ] #6 Focused and broader automated tests, CSS generation/parity, static checks, and diff checks pass or any inherited baseline failure is reproduced on the untouched base with exact evidence.
<!-- AC:END -->

## Decision Record Check

ADR required: no new ADR
ADR paths: `backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md`, `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`, `backlog/decisions/035-file-notes-session-git-index-controls.md`, `backlog/decisions/038-file-notes-guarded-session-commit.md`, `backlog/decisions/039-file-notes-guarded-session-push.md`, `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`
Reason: this closes verification and documentation for already-decided behavior without adding architecture.
