---
id: TASK-985
title: Add session-scoped Git status and staging to File Notes
status: To Do
assignee: []
created_date: '2026-07-27 19:52'
updated_date: '2026-07-27 20:01'
labels:
  - notes
  - git
  - library
  - ux
  - security
dependencies:
  - TASK-969
  - TASK-982
documentation:
  - Docs/superpowers/specs/2026-07-27-file-notes-session-git-staging-design.md
  - backlog/decisions/034-file-notes-session-git-index-controls.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users inspect and safely stage or unstage only paths changed during the current Chatbook File Notes session while preserving disk authority and existing external Git index state. This first Git slice must remain optional and must not add commit, push, remote, branch-mutation, or repository-wide status behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A selected File Notes root inside one supported Git worktree shows coalesced current-session path rows with actual Git state and explicit Session paths only and whole-file staging labels.
- [ ] #2 User can stage or unstage one eligible logical row and can Stage All or Unstage All eligible rows; move endpoints remain one inseparable group and pending autosave is flushed before staging.
- [ ] #3 Stage and unstage never modify note bytes, the File Notes SQLite replica, unrelated index entries, or File Notes session history.
- [ ] #4 Missing or unsupported Git, non-repository or replaced roots, index locks, command failures, and status timeouts remain nonfatal and never disable File Notes editing.
- [ ] #5 The Session Git navigator view, per-row and bulk controls, refresh behavior, and trust confirmation remain responsive without remounting the editor or breaking narrow-terminal navigation.
- [ ] #6 Focused unit, disposable-repository integration, mounted Textual, and disposable-repository acceptance checks cover the approved safety boundary without requiring full-suite or network execution.
- [ ] #7 Commit, push, remotes, credentials, branch mutation, hunk staging, repository initialization, persistent staging ownership, full-repository status, and nested-repository management remain absent.
- [ ] #8 A fresh preflight blocks observed pre-existing or partially staged same-path content, conflicts, ignored paths, and nested repository boundaries without intentionally targeting that index state.
- [ ] #9 Chatbook unstages only exact index entries it staged in the current process while repository, HEAD, endpoint topology, and index signatures still match; observed external changes or a new process revoke ownership.
- [ ] #10 Concurrent external index mutation during one Chatbook Stage or Unstage action is an unsupported race; lock contention and observable uncertainty are surfaced without claiming atomic cross-process ownership.
<!-- AC:END -->
