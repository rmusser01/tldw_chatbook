---
id: TASK-985
title: Add session-scoped Git status and staging to File Notes
status: To Do
assignee: []
created_date: '2026-07-27 19:52'
updated_date: '2026-07-27 20:26'
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
Let users inspect and safely stage or unstage only paths changed during the current Chatbook File Notes session while preserving disk authority and external Git index state observed by a fresh preflight. This first Git slice remains optional and excludes commit, push, remote, branch-mutation, and repository-wide status behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A selected File Notes root inside one supported Git worktree replaces the unbounded session summary with Session Git (N), where N is the coalesced current-session group count; after trust, rows show actual path-scoped Git state plus explicit Session paths only and whole-file staging labels.
- [ ] #2 User can Stage or Unstage one eligible logical row and can Stage All or Unstage All eligible rows; pending autosave is flushed, move lineage remains one policy group, and only effective Git-matchable mutation paths are supplied with create, modify, and delete semantics.
- [ ] #3 Chatbook selects no Git operation intended to update the worktree and never writes the File Notes SQLite replica or session history during Git actions; unrelated index entries are not intentionally targeted, while arbitrary side effects from explicitly trusted user-configured filters are disclosed as outside the guarantee.
- [ ] #4 Missing, unsupported, untrusted, locked, unsafe, non-repository, or replaced Git roots plus command failures and status timeouts remain nonfatal and never disable File Notes editing.
- [ ] #5 Chatbook reverses only index entries it staged in the current process while repository, HEAD, mode, object ID, stage, semantic flags, and approved no-op move-endpoint preconditions still match; observed external changes or a new process revoke Unstage eligibility.
- [ ] #6 Concurrent external index mutation during one Chatbook Stage or Unstage action remains an explicitly unsupported race; lock contention and observable uncertainty are surfaced without claiming atomic cross-process ownership.
- [ ] #7 Focused unit, disposable-repository integration, mounted Textual, one 1,000-plus-unrelated-notes scale fixture, and disposable-repository acceptance checks cover the approved boundary without full-suite, network, pagination, or broad performance execution.
- [ ] #8 Commit, push, pull, fetch, remotes, credentials, branch mutation, hunk staging, repository initialization or repair, persistent staging ownership, full-repository status, and nested-repository management remain absent.
- [ ] #9 The documented state-to-action mapping controls selected and bulk eligibility; refresh preserves stable row selection, performs no periodic or hidden-view Git work, permits one status query plus at most one coalesced rerun, makes mutations wait for active status, and keeps editor input and narrow-terminal navigation responsive.
- [ ] #10 A fresh preflight blocks observed pre-existing or partially staged same-path state, conflicts, ignored paths, nested repository boundaries, directory or other unsafe worktree types, and nondefault semantic index flags such as skip-worktree, assume-unchanged, or intent-to-add.
- [ ] #11 Before the first worktree-aware status or Stage for a selected root and repository identity, Chatbook explains that configured filters may execute and requires process-lifetime trust; identity is revalidated before each worktree-aware command, identity change clears trust, and declining runs neither operation while File Notes remains usable.
<!-- AC:END -->
