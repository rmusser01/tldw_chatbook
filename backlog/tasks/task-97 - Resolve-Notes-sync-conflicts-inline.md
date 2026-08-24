---
id: TASK-97
title: Resolve Notes sync conflicts inline
status: Done
assignee:
  - '@codex'
created_date: '2026-06-11 17:05'
updated_date: '2026-08-24 02:33'
labels: []
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-08-22-task-97-notes-sync-conflict-resolution-design.md
  - Docs/superpowers/plans/2026-08-22-task-97-notes-sync-conflict-resolution.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the retired ASK-engine modal concept with reviewed inline conflict resolution in the lasting Database Notes sync flow. Users compare the current file and Library note, stage Keep file, Keep note, Keep both, or Skip for now without mutation, then apply an explicitly reviewed subset through the existing durable runtime and recovery journal. Conflict outcomes remain device-local, recoverable, and visible in bounded resolution history; deletion review remains outside this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Only eligible bound content-change conflicts appear in the existing inline reviewed-sync flow with a bounded file/note comparison and Keep file, Keep note, Keep both, and Skip for now choices; identity, move, representation, creation, deletion, and duplicate-authority conflicts remain blocked
- [x] #2 Staging or changing a choice performs no mutation; selections survive paging but are discarded when the review becomes stale or is abandoned
- [x] #3 Apply reviewed revalidates fresh authority and applies safe actions plus explicitly selected conflict resolutions; skipped conflicts remain Needs attention
- [x] #4 Keep file and Keep note resolve only that occurrence without changing the root's configured direction
- [x] #5 Keep both durably preserves the original note content in a new unbound manual conflict copy before updating the original bound note, supports restart recovery, and records both outcomes
- [x] #6 Each completed resolution leaves an at-action per-item receipt with Undo and Dismiss, durable resolution history records the explicit choice, and Undo is offered only while exact 30-day recovery remains valid; later edits are never overwritten
- [x] #7 Deletion, pause, managed-placement, capability, and activation attention remain blocked and unchanged
- [x] #8 Comparison content and private authority never enter persistent diagnostics or logs, and unsupported filesystem writes remain fail-closed
- [x] #9 The inline choices, comparison, receipts, and history are keyboard reachable, communicate state without color alone, preserve focus during staged updates, and remain usable at 60x20
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute the reviewed plan in
`Docs/superpowers/plans/2026-08-22-task-97-notes-sync-conflict-resolution.md`
task by task, preserving the existing runtime, executor, device-state, Library
controller, and Textual ownership boundaries described in the approved design.

ADR required: no new ADR.

Existing ADRs:

- `backlog/decisions/055-library-destructive-action-reversibility-rule.md`
- `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`
- `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`

Reason: this task directly implements those existing recovery, ownership,
privacy, and round-trip decisions without introducing a new persistence or
authority boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Implemented exact conflict eligibility, bounded Note-to-File comparison, mutation-free staging, subset Apply, occurrence-only Keep file/Keep note, restart-safe Keep both, 30-day per-item Undo, receipts, and bounded durable history through the existing runtime/executor/device-state ownership chain.
- Extended the retained Library controller and Textual canvas with keyboard- and non-color-reachable inline controls at wide and 60x20 sizes; updated the user guide and added the mixed-partial-restart verification lesson.
- Preserved deletion/capability/managed-placement blockers, automatic sync direction, recovery-before-write, private projections, and filesystem confinement. No dependency, schema, database owner, modal, or new ADR was introduced; ADRs 055, 059, and 073 govern the implementation.
- Follow-up gates prevent no-op-only Apply, avoid conflict-free duplicate observation, keep persisted-root history reachable through empty/read-failure states, delete the transient history-availability probe, and enforce exact shipped labels. The final boundary correction centralizes the four runtime-admitted manual action kinds, submits only those IDs plus exact reviewed NO_CHANGE IDs, and excludes MOVE_FILE/blocked kinds without blocking legitimate mixed work. Unknown IDs still fail closed.
- Evidence: corrected exact 12-file matrix **747 passed, 8 warnings in 168.93s**; scoped Ruff/format, compileall, CSS generation, private-SQLite/backup/legacy/startup/privacy governance, and diff checks passed. Exact MyPy output remains the known baseline of 144 errors in 5 files, and the shared models module passes standalone MyPy. Isolated scratch-profile TUI UAT proved all choices, partial Skip, receipt Dismiss/Undo, restart history, and shared config/DB fingerprints unchanged before/after.
<!-- SECTION:NOTES:END -->
