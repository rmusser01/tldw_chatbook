---
id: TASK-1350
title: Add guarded session commit to File Notes
status: To Do
assignee: []
created_date: '2026-07-29 19:17'
labels:
  - notes
  - git
  - library
  - ux
  - security
dependencies:
  - TASK-1213
  - TASK-1235
documentation:
  - >-
    Docs/superpowers/specs/2026-07-29-file-notes-guarded-session-commit-design.md
  - backlog/decisions/038-file-notes-guarded-session-commit.md
  - backlog/decisions/035-file-notes-session-git-index-controls.md
  - backlog/decisions/033-application-session-state-ownership.md
  - backlog/decisions/029-file-notes-disk-authority.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users turn the exact Chatbook-owned staged notes from the current File Notes session into one reviewed local Git commit without absorbing unrelated staged content, changing file/SQLite authority, or expanding into push and remote management.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Prepare session for commit exposes Commit staged (N), a required-subject/optional-body form, a read-only review, and an explicit Confirm commit step with unambiguous keyboard, focus, cancellation, and 40x20 behavior.
- [ ] #2 Immediately before execution, Chatbook proves that the complete staged delta against the captured attached branch HEAD exactly equals current Chatbook staging ownership; unrelated staged content, newer unstaged edits on included groups, conflicts, special index states, detached or unborn HEAD, and active Git sequencer operations block without starting a commit child or making a Chatbook-authored HEAD, index, note-worktree, or SQLite mutation, subject to ADR-035's disclosed trusted-filter boundary.
- [ ] #3 The review shows the exact normalized UTF-8 message, branch, old commit, included-note count/list, resolved author and committer, and the policy that hooks are bypassed and the commit is unsigned; missing identity or invalid message blocks.
- [ ] #4 Confirm revalidates the immutable review snapshot under the process owner mutation gate, keeps the editor read-only through terminal outcome, executes one direct-argv noninteractive normal commit with hooks and signing disabled, and never abandons the retained child lifecycle.
- [ ] #5 Postflight classifies Cancelled, Blocked, Succeeded, Failed unchanged, or Uncertain from the child result plus exact branch, parent, full-tree, metadata, and logical-index facts; Chatbook never rolls back, deletes locks, or automatically retries.
- [ ] #6 A proven success retires only fully committed session groups, retains newer worktree edits, refreshes actual status, and reports the committed note count plus the promise that unrelated changes were untouched; failure and uncertainty preserve the message draft and provide exact recovery.
- [ ] #7 Commit actions never rewrite note bytes or mutate the independent SQLite replica, revisions, tombstones, repository configuration, remotes, credentials, push/pull/fetch, amend, signing, or general branch management; the feature adds no durable trust state and preserves ADR-035 trust invalidation semantics.
- [ ] #8 Focused unit, disposable-repository integration, lifecycle, and mounted Textual tests include automated 1,000-note presentation coverage and prove a bounded Git process count; a representative two-note flow receives narrow/wide live UAT, with no unrelated full-suite or broad CI expansion.
<!-- AC:END -->
