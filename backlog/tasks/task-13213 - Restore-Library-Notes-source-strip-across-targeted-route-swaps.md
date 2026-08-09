---
id: TASK-13213
title: Restore Library Notes source strip across targeted route swaps
status: Done
assignee:
  - '@codex'
created_date: '2026-08-09 19:57'
updated_date: '2026-08-09 13:43'
labels:
  - library
  - notes
  - ux
dependencies:
  - TASK-1411
  - TASK-2850
documentation:
  - backlog/decisions/027-portable-database-note-session-coordinator.md
  - backlog/decisions/029-file-notes-disk-authority.md
priority: high
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore reliable access to the Library Notes source selector so users can reach and safely leave file-backed notes after navigating into Notes from another Library route, without discarding the safe targeted-refresh optimization for structurally compatible routes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Entering Library Notes from a route without Notes-specific chrome visibly mounts the Database and Files choices at both 120x40 and 160x45, and choosing Files opens the retained file-notes workspace.
- [x] #2 Leaving Library Notes removes the Notes-only source strip instead of leaving stale controls on other Library routes.
- [x] #3 Targeted canvas replacement remains available when the mounted contextual chrome matches the destination route.
- [x] #4 Reselecting Notes while Files owns the canvas retains the file-notes workspace, and Escape returns from Files to Database Notes.
- [x] #5 Switching from a loading Database Note to Files invalidates the pending database session so its late completion cannot repopulate hidden editor state.
- [x] #6 Focused mounted UI regressions, linting, changed-hunk formatting review, and diff checks pass.
- [x] #7 Selecting Create Note from Files mode returns Notes to the Database source and restores the Database Notes Escape action.
- [x] #8 Notes-source and Notes-canvas control flow uses named constants instead of repeated behavior-defining string literals.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: N/A; this fix conforms to `backlog/decisions/027-portable-database-note-session-coordinator.md` and `backlog/decisions/029-file-notes-disk-authority.md`.

Reason: This is a contained UI composition-lifecycle regression and does not change storage ownership, sync policy, schemas, security boundaries, or cross-module interfaces.

1. Add mounted regression coverage for entering and leaving Notes across the contextual source-strip boundary.
2. Gate targeted Library canvas replacement on whether the mounted Notes-specific chrome matches the destination route.
3. Preserve the existing File Notes exit and database-session invalidation contracts once the source selector is reachable again.
4. Run focused Library Notes and shell tests at the supported compact terminal sizes.
5. Run lint, changed-hunk formatting review, diff, and self-review checks, then document the implementation outcome.
6. Address Qodo's source-state and maintainability findings with a Files-to-Create regression and named behavior constants.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a contextual-chrome guard to the targeted Library canvas replacement. Crossing the Notes boundary now awaits the canonical full recompose so the Database/Files source strip is added or removed with its route.
- Preserved targeted replacement for compatible Database Notes routes, detaching the hidden outgoing canvas before mounting a stable-ID replacement to avoid Textual duplicate-ID rejection. File Notes remains on the full-composition path so its retained workspace is never replaced by a database canvas.
- Restored File Notes lifecycle contracts exposed once the selector became reachable: Database Notes shortcuts no longer claim Escape in Files mode, and switching to Files invalidates/closes any pending database-note session before a late load can commit.
- Updated mounted regressions for Notes/Media contextual chrome, file-workspace retention, keyboard source selection at 120x40 and 160x45, Escape, shell framing, and late-load invalidation. Seven consolidated mounted cases, two focused binding/action cases, and the 50-route-cycle ownership stress case passed.
- Targeted Ruff (with seven unrelated pre-existing E721 findings excluded) and `git diff --check` passed. Changed hunks were reviewed against Ruff's formatter output; the three large pre-existing UI files still report whole-file formatting drift, so they were not mechanically reformatted into an unrelated large diff.
- The existing Library user guide already accurately documents `Database | Files`; no user-facing documentation change was required. Added the route-owned-sibling incident to `backlog/docs/lessons-testing-evidence.md`.
- ADR required: no. The change conforms to ADR-027 and ADR-029 and does not alter storage, synchronization, or ownership boundaries.
- Addressed both actionable Qodo findings: Create Note now normalizes Files to the Database source only after File Notes flush and transition admission succeed, and routing/source checks now share named source and canvas-kind constants. The new transition regression asserts the selected route, source, Escape action, and lease ordering.
- Focused Qodo follow-up verification passed: eight route/source/session tests, targeted Ruff with the file's pre-existing E721 findings excluded, and `git diff --check`. The owner directed this PR to ignore the unrelated duplicate TASK-3401 CI failure, so no unrelated backlog task was renumbered.
<!-- SECTION:NOTES:END -->
