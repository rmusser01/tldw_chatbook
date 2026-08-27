---
id: TASK-22513
title: Polish Library Notes work-first editors and Folder Files shell
status: Done
assignee:
  - '@codex'
created_date: '2026-08-27 06:04'
labels:
  - library
  - notes
  - ui
dependencies:
  - TASK-22032
  - TASK-19001
references:
  - Docs/superpowers/specs/2026-08-26-library-notes-ux-improvements-design.md
  - backlog/decisions/086-library-adaptive-reader-shell.md
  - backlog/decisions/076-library-lifecycle-progressive-disclosure.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make both Library Notes authorities calmer and more efficient for sustained writing while preserving database and on-disk authority, state, recovery, and every incumbent capability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Database Notes and Folder Files use the shared adaptive reader shell as their only geometry and focus-evacuation owner, including an independently collapsible Folder Files tree.
- [x] #2 Work-first Library collapse activates once per approved Notes work session, manual expansion wins, responsive changes never persist, and all reset predicates are deterministic.
- [x] #3 Database list and Folder tree visibility and width preferences use independent normalized keys, Settings controls, environment overrides, and race-safe persistence authorities.
- [x] #4 Database Notes retains Edit, Preview, Info, and all navigator workflows; Folder Files exposes Edit and Manage, retains autosave and recovery, and gains neither Markdown Preview nor a manual Save control.
- [x] #5 Only the two note-body editors retain their resting background on focus and use a geometry-stable heavy outline with verified theme contrast.
- [x] #6 Primary headers preserve authority and consequential status, apply the approved status precedence, and keep safe recovery actions visible without truncation.
- [x] #7 The Notes-specific Ctrl+S binding and hints are removed without adding a replacement shortcut; visible Save remains keyboard reachable.
- [x] #8 Targeted reducer, configuration, Settings, shell, Notes, Folder Files, accessibility, CSS, and live isolated TUI verification pass.
- [x] #9 Library Notes and Folder Files user documentation reflects the final pane, mode, focus, save, and shortcut behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan: Docs/superpowers/plans/2026-08-26-library-notes-work-first-ux.md
Design: Docs/superpowers/specs/2026-08-26-library-notes-ux-improvements-design.md
ADR required: no
ADR path: N/A
Reason: this directly implements ADR-086's shared adaptive-reader shell and ADR-076's progressive-disclosure boundaries without changing storage, sync/conflict policy, service ownership, security, dependencies, or cross-module contracts.
Existing ADRs: backlog/decisions/086-library-adaptive-reader-shell.md; backlog/decisions/076-library-lifecycle-progressive-disclosure.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Reused `LibraryAdaptiveReaderShell` for both Notes authorities, with independent Database list and Folder Files tree preferences, Settings/environment integration, drag widths, responsive behavior, and focus restoration. Added the session reducer that applies work-first collapse once per admitted editing session while preserving manual overrides and deterministic resets.
- Reorganized Database Notes into Edit/Preview/Info and Folder Files into Edit/Manage without removing incumbent workflows. Kept Folder Files autosave-only, kept Database Save visible and keyboard reachable, removed only the Notes Ctrl+S binding/hints, and documented both authorities.
- Limited boundary-only focus styling to the two note-body editors. Verified unchanged fill and geometry across every shipped theme with at least 3:1 boundary contrast. Preserved specific validation, save-failure, conflict, and blocked-action detail in the painted Notes status channel.
- Hardened New/Move path-task admission across root, session, and authority transitions. Review found and fixed the production-order shared-save race by cancelling a path task admitted during flush before source-transition admission.
- Targeted automated evidence included a final 162-case Notes-focused integration batch, a 14-case task-regression batch, and a post-review 12-case race/status/state batch, all passing. Earlier task-focused batches covered Settings, Git/maintenance disclosure, CSS source/build parity, and the shared shell. Ruff, task-owned format checks, compileall, and branch/worktree diff checks passed.
- An isolated live Textual walkthrough passed at 160x40, 120x35, 119x35, 100x30, 80x24, 79x24, and 60x20. It exercised both authorities, one-time collapse/manual reopen, resize and authority round trips, drafts/cursor/undo/tree/search retention, modes, visible actions, focus styling, and shortcut scope. All data/config paths were redirected to `/private/tmp/task22513-live.3ieSLy`; the real config hash remained `1b23f3a533632631678644eaeabcb1c5737e2cb7b9d6dc1d6e9323f622fece12` before and after.
- Independent spec review approved the branch. Quality review identified two P2 issues; both were fixed with regression coverage, and the final re-review approved the result.
- The broader planned matrix was interrupted after review fixes made its collected code stale. Its task-owned failures were rerun green. Unrelated Settings ownership, legacy Library hub, nested `.venv` CSS-walker, and formatting failures reproduce on the detached pre-feature base or clean-worktree comparison and were not changed. The full repository suite was not run because the user did not opt into a full sweep.
- ADR required: no. ADR path: N/A. This implements existing ADR-086 and ADR-076 without changing storage, sync/conflict policy, service contracts, security, dependencies, or ownership boundaries.
