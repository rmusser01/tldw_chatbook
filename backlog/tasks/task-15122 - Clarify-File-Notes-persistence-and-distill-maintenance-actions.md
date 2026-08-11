---
id: TASK-15122
title: Clarify File Notes persistence and distill maintenance actions
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 14:32'
updated_date: '2026-08-11 15:08'
labels:
  - notes
  - library
  - ux
  - accessibility
dependencies:
  - TASK-14879
  - TASK-14880
  - TASK-14904
references:
  - >-
    .impeccable/critique/2026-08-11T06-03-15Z__ok-widgets-library-library-file-notes-workspace-py.md
documentation:
  - backlog/decisions/011-chatbook-workbench-ui-system.md
  - backlog/decisions/029-file-notes-disk-authority.md
  - backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md
  - backlog/decisions/035-file-notes-session-git-index-controls.md
priority: medium
type: enhancement
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the File Notes editor state where normal edits are saved, keep recovery instructions complete in compact terminals, reduce peer file actions through progressive disclosure, and introduce Session Git through the outcome users seek. Preserve disk authority, replica behavior, Git scope, and every existing operation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every normal persistence state explicitly identifies the linked local folder as the save authority, and recovery copy distinguishes preserved drafts from ordinary autosave.
- [x] #2 Warning, error, and recovery instructions remain complete and keyboard reachable at 160x45, 120x40, and 40x20; routine Git telemetry may remain compact.
- [x] #3 The primary editor row exposes only frequent actions, while all existing secondary file operations remain available through one keyboard-operable Maintenance disclosure with safe focus repair.
- [x] #4 The Git entry and introductory copy lead with Review session changes and explain that only notes changed during the current Chatbook session are reviewed and committed.
- [x] #5 Focused mounted regressions, compact-terminal tests, targeted Ruff, Python compilation, diff checks, and self-review pass without changing disk, replica, session-owner, staging, commit, or push behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; conform to `backlog/decisions/011-chatbook-workbench-ui-system.md`, `backlog/decisions/029-file-notes-disk-authority.md`, `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`, and `backlog/decisions/035-file-notes-session-git-index-controls.md`.
Reason: this is a presentation, copy, disclosure, focus, and compact-layout refinement within existing disk and Git authority boundaries.

1. Add mounted regressions for explicit local-folder autosave states, recovery-copy visibility, the maintenance disclosure, and the outcome-led Git entry copy.
2. Clarify persistence and recovery labels without changing autosave, conflict, replica, or disk behavior.
3. Move secondary file actions behind one keyboard-operable Maintenance disclosure with deterministic visibility and focus repair.
4. Let critical Git recovery copy wrap fully while keeping routine telemetry compact, and lead the Git entry with Review session changes.
5. Run focused mounted tests, static checks, compact-terminal verification, self-review, and close the Backlog task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Clarified normal autosave states so the linked local folder is visibly authoritative and renamed the recovery operation to `Save draft as copy`. Kept frequent editor actions in the primary row and moved Move, Protect, Reload, and Refresh behind one keyboard-operable Maintenance disclosure with compact two-column layout and focus repair.

Renamed the Git entry and panel to `Review session changes`, described its current-session scope, and let warning/error/recovery text wrap completely in the scrollable panel while retaining two-line fitting for routine telemetry. Disk, replica, session-owner, staging, commit, and push behavior were unchanged.

Verification: 50 File Notes workspace tests passed; 19 focused Git/compact-terminal tests passed; 109 shared focus/CSS integrity tests passed; final five-test copy/disclosure regression selection passed; targeted Ruff, Python compilation, and `git diff --check` passed. The full Git module remains at 144 passed and one documented pre-existing baseline failure in the unrelated Protect-button stage gate (`test_stage_flushes_then_gate_keeps_editor_back_and_one_latest_refresh`).

ADR required: no. The implementation conforms to ADR-011, ADR-029, ADR-031, and ADR-035. Modified the File Notes workspace and Git panel, their focused regression suites, and the Impeccable critique artifact.

Qodo follow-up: added Google-style parameter documentation for the complete-copy APIs; relaxed the stacked delete-confirmation toolbar so all dirty-state controls remain contained at 40×20; and removed retired `Prepare session` / `Session Git mutation` wording from user-visible recovery, busy, and trust copy. The review regression was reproduced before the layout fix, then the six focused mounted cases and the complete 51-test workspace suite passed. Targeted Ruff, Python compilation, and diff checks also passed.
<!-- SECTION:NOTES:END -->
