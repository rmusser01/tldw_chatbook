---
id: TASK-14904
title: Distill File Notes Session Git action hierarchy
status: Done
assignee:
  - '@codex'
created_date: '2026-08-10 19:30'
updated_date: '2026-08-10 20:57'
labels:
  - notes
  - git
  - library
  - ux
  - accessibility
priority: medium
dependencies:
  - TASK-1235
  - TASK-1350
  - TASK-1711
references:
  - .impeccable/critique/2026-08-10T06-12-44Z__ok-widgets-library-library-file-notes-workspace-py.md
documentation:
  - backlog/decisions/011-chatbook-workbench-ui-system.md
  - backlog/decisions/029-file-notes-disk-authority.md
  - backlog/decisions/035-file-notes-session-git-index-controls.md
  - backlog/decisions/038-file-notes-guarded-session-commit.md
  - backlog/decisions/039-file-notes-guarded-session-push.md
---

## Description

Make the File Notes Session Git list surface reveal only the controls relevant to its current trust, selection, staging, commit, and push state. Reduce persistent operational clutter while preserving every existing session-only Git authority, recovery path, keyboard workflow, and compact-terminal guarantee.

## Acceptance Criteria

- [x] Before repository trust, the list surface keeps Back and Trust visible while hiding trusted-only refresh, row mutation, bulk, commit, and push actions; after trust, Trust disappears and Refresh becomes available.
- [x] A selected session-note row exposes only its one applicable Stage, Stage update, or Unstage action, and focus moves to a safe visible owner whenever state removes the focused action.
- [x] Bulk Stage/Unstage controls sit behind one compact, keyboard-operable disclosure, show accurate eligible counts, remain collapsed by default, and disappear when neither bulk action can apply.
- [x] Commit appears only when owned staged session notes are eligible for review, and guarded Push appears only for the exact qualifying local commit candidate; blocked or unavailable states retain explicit recovery/status copy instead of disabled ghost controls.
- [x] Back, Refresh, trust, row, bulk, commit, push, mutation-progress, and recovery workflows remain keyboard-operable with complete labels and stable layout at 160x45, 120x40, and 40x20.
- [x] Mounted regressions, focused Session Git/workspace tests, targeted Ruff, Python compilation, diff checks, and self-review pass without changing Git authority, repository mutation, disk, replica, session-owner, commit, or push behavior.

## Implementation Plan

ADR required: no
ADR path: N/A; conform to `backlog/decisions/011-chatbook-workbench-ui-system.md`, `backlog/decisions/029-file-notes-disk-authority.md`, `backlog/decisions/035-file-notes-session-git-index-controls.md`, `backlog/decisions/038-file-notes-guarded-session-commit.md`, and `backlog/decisions/039-file-notes-guarded-session-push.md`.
Reason: this task changes only list-level action visibility, disclosure, focus repair, and responsive presentation inside the established Session Git workflow; it adds no storage, synchronization, authority, service, commit, push, or platform policy.

1. Characterize the current mounted trust, row-action, bulk, commit, and push projections and add failing tests for the intended progressive hierarchy at wide, moderate, and compact sizes.
2. Add one retained bulk-action disclosure and derive every list-level control's visibility from the existing trusted status, selected row, eligibility counts, commit projection, and guarded-push candidate.
3. Keep blocked and zero-state explanations in concise status copy, repair focus whenever disclosure or state hides its owner, and preserve all existing message/event boundaries.
4. Mutation-check the new visibility and focus guards, then run focused Session Git/workspace suites and static/diff checks.
5. Self-review against the Impeccable critique and ADR boundaries, record verification evidence, complete the acceptance criteria, and move TASK-14904 to Done.

## Implementation Notes

- Added a retained `Show/Hide bulk` disclosure with live Stage/Unstage eligibility counts; bulk actions are hidden by default and only the non-zero actions render when expanded.
- Reduced each selected row to one policy-owned next action (`Stage`, `Stage update`, or `Unstage`), hid zero-count Commit and transiently unavailable Push actions, and kept explanatory zero/status copy visible.
- Preserved Back and Refresh as stable list controls, repaired focus after disappearing row/bulk actions without stealing outside focus, and withheld mutations until the current retained row generation is mounted.
- Updated mouse, keyboard, compact-terminal, commit, push, focus, and mutation-lifetime tests. Deferred-focus assertions now wait for the existing phase-safe callback, and retained-row action helpers wait for the mounted projection before acting.
- Verification: 142 File Notes Session Git tests passed with the separately passing 1,000-row case; the one excluded failure (`test_stage_flushes_then_gate_keeps_editor_back_and_one_latest_refresh`) is identical on the pre-task baseline and concerns the unrelated Protect-button gate. After Qodo follow-up, guarded-push tests passed 60/60, shared focus/CSS integrity tests passed 109/109, and the task-focused matrix passed 29/29. Ruff, Python compilation, and `git diff --check` passed.
- ADR required: no. The implementation stays within ADR-011, ADR-029, ADR-035, ADR-038, and ADR-039; no Git authority, service, disk, replica, commit, or push boundary changed.
- Added a testing lesson documenting why owner row state is not proof that retained `ListView` rows are mounted and why `Pilot.pause()` is unsafe during child teardown.
- Qodo review follow-up centralized the retained-row polling budget/interval as named test constants and made Push-to-list focus repair fall back through visible, enabled Refresh/Back controls when mutation hides the Push action; a mounted regression covers the mutating return path.
